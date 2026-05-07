import torch
import numpy as np
import pandas as pd
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Optional
from qts_core.logger import logger

class ModalityPlugin(ABC):
    @abstractmethod
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor: pass
    @property
    @abstractmethod
    def name(self) -> str: pass

class ModalityRegistry:
    _plugins: Dict[str, Type[ModalityPlugin]] = {}
    @classmethod
    def register(cls, name: str, plugin_cls: Type[ModalityPlugin]):
        cls._plugins[name] = plugin_cls
        logger.info(f"Registered modality plugin: {name}")
    @classmethod
    def get_plugin(cls, name: str, **kwargs) -> Optional[ModalityPlugin]:
        plugin_cls = cls._plugins.get(name)
        return plugin_cls(**kwargs) if plugin_cls else None
    @classmethod
    def list_registered(cls) -> List[str]: return list(cls._plugins.keys())
    @classmethod
    def create_all(cls, config: Optional[dict] = None) -> List[ModalityPlugin]:
        instances = []
        # Support both 'plugins' block and default auto-discovery
        enabled_plugins = config.get('enabled_plugins', cls.list_registered()) if config else cls.list_registered()
        for name in enabled_plugins:
            plugin_cls = cls._plugins.get(name)
            if plugin_cls:
                plugin_config = config.get('plugin_settings', {}).get(name, {}) if config else {}
                # Pass global signal physics if relevant
                if config and 'signal_physics' in config:
                    plugin_config['signal_config'] = config['signal_physics']
                instances.append(plugin_cls(**plugin_config))
        return instances

def register_modality(name: str):
    def decorator(cls: Type[ModalityPlugin]):
        ModalityRegistry.register(name, cls); return cls
    return decorator

@register_modality("x_seq")
class SequentialPlugin(ModalityPlugin):
    def __init__(self, **kwargs):
        from research_lab.alpha_core import FractionalDifferencer
        d = kwargs.get('signal_config', {}).get('fractional_differentiation', {}).get('d_param', 0.4)
        self.fd = FractionalDifferencer(d=d); self._name = "x_seq"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        windows = [stationary[t-lookback:t].reshape(-1, 1) for t in range(lookback, len(stationary) + 1)]
        return torch.tensor(np.array(windows)).float()

@register_modality("x_spatial")
class SpatialPlugin(ModalityPlugin):
    def __init__(self, **kwargs):
        from research_lab.alpha_core import WaveletFeatureGenerator, FractionalDifferencer
        sig = kwargs.get('signal_config', {})
        d = sig.get('fractional_differentiation', {}).get('d_param', 0.4)
        
        # SENIOR FIX: Ensure scales are correctly extracted from config
        wavelet_config = sig.get('wavelet_transform', {})
        sc = wavelet_config.get('scales', 2**np.arange(1, 9))
        wv = wavelet_config.get('wavelet', 'mexh')
        
        logger.info(f"SpatialPlugin: Initializing with {len(sc)} scales.")
        
        self.fd = FractionalDifferencer(d=d)
        self.wfg = WaveletFeatureGenerator(scales=np.array(sc), wavelet=wv); self._name = "x_spatial"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        spectrogram = self.wfg.generate(pd.Series(stationary))
        windows = [spectrogram[:, t-lookback:t] for t in range(lookback, len(stationary) + 1)]
        return torch.tensor(np.array(windows)).unsqueeze(1).float()

@register_modality("x_graph")
class GraphPlugin(ModalityPlugin):
    def __init__(self, **kwargs):
        from research_lab.alpha_core import FractionalDifferencer
        d = kwargs.get('signal_config', {}).get('fractional_differentiation', {}).get('d_param', 0.4)
        self.fd = FractionalDifferencer(d=d); self._name = "x_graph"
        self.feature_dim = kwargs.get('feature_dim', 8)
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        windows = []
        for t in range(lookback, len(stationary) + 1):
            start = max(0, t - self.feature_dim)
            node_feat = stationary[start:t]
            if len(node_feat) < self.feature_dim: node_feat = np.pad(node_feat, (self.feature_dim - len(node_feat), 0))
            windows.append(node_feat)
        return torch.tensor(np.array(windows)).float()

@register_modality("x_volume")
class VolumePlugin(ModalityPlugin):
    """Encodes normalized volume dynamics as a distinct modality."""
    def __init__(self, **kwargs): self._name = "x_volume"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        if 'volume' not in pit_view.columns: return torch.zeros((len(pit_view)-lookback+1, lookback, 1))
        
        v = pit_view['volume'].values
        # SENIOR FIX: If volume is already standardized (contains negative values), skip log1p
        if np.any(v < 0):
            v_norm = v
        else:
            # Log-transform and Z-score raw volume
            v_log = np.log1p(v)
            v_norm = (v_log - np.mean(v_log)) / (np.std(v_log) + 1e-9)
            
        windows = [v_norm[t-lookback:t].reshape(-1, 1) for t in range(lookback, len(v_norm) + 1)]
        return torch.tensor(np.array(windows)).float()
