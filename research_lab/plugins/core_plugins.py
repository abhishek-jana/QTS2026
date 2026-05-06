import torch
import numpy as np
import pandas as pd
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
    """Registry for modality plugins to enable auto-discovery."""
    _plugins: Dict[str, Type[ModalityPlugin]] = {}

    @classmethod
    def register(cls, name: str, plugin_cls: Type[ModalityPlugin]):
        cls._plugins[name] = plugin_cls
        logger.info(f"Registered modality plugin: {name}")

    @classmethod
    def get_plugin(cls, name: str, **kwargs) -> Optional[ModalityPlugin]:
        plugin_cls = cls._plugins.get(name)
        if plugin_cls:
            return plugin_cls(**kwargs)
        return None

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._plugins.keys())

    @classmethod
    def create_all(cls, config: Optional[dict] = None) -> List[ModalityPlugin]:
        """Creates instances of all registered plugins, potentially filtered/configured by config."""
        instances = []
        enabled_plugins = config.get('enabled_plugins', cls.list_registered()) if config else cls.list_registered()
        
        for name in enabled_plugins:
            plugin_cls = cls._plugins.get(name)
            if plugin_cls:
                # Here we could pass specific config for each plugin if needed
                plugin_config = config.get('plugin_settings', {}).get(name, {}) if config else {}
                instances.append(plugin_cls(**plugin_config))
        return instances

def register_modality(name: str):
    def decorator(cls: Type[ModalityPlugin]):
        ModalityRegistry.register(name, cls)
        return cls
    return decorator

@register_modality("x_seq")
class SequentialPlugin(ModalityPlugin):
    def __init__(self, d_param: float = 0.4):
        from research_lab.alpha_core import FractionalDifferencer
        self.fd = FractionalDifferencer(d=d_param); self._name = "x_seq"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        windows = [stationary[t-lookback:t].reshape(-1, 1) for t in range(lookback, len(stationary) + 1)]
        return torch.tensor(np.array(windows)).float()

@register_modality("x_spatial")
class SpatialPlugin(ModalityPlugin):
    def __init__(self, scales: np.ndarray = None):
        from research_lab.alpha_core import WaveletFeatureGenerator, FractionalDifferencer
        self.fd = FractionalDifferencer(d=0.4); self.scales = scales if scales is not None else 2 ** np.arange(1, 9)
        self.wfg = WaveletFeatureGenerator(scales=self.scales); self._name = "x_spatial"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        spectrogram = self.wfg.generate(pd.Series(stationary))
        windows = [spectrogram[:, t-lookback:t] for t in range(lookback, len(stationary) + 1)]
        return torch.tensor(np.array(windows)).unsqueeze(1).float()

@register_modality("x_graph")
class GraphPlugin(ModalityPlugin):
    def __init__(self, feature_dim: int = 8):
        from research_lab.alpha_core import FractionalDifferencer
        self.fd = FractionalDifferencer(d=0.4); self._name = "x_graph"; self.feature_dim = feature_dim
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        windows = []
        for t in range(lookback, len(stationary) + 1):
            # FIX: Ensure we don't index before the start of the stationary array
            start = max(0, t - self.feature_dim)
            node_feat = stationary[start:t]
            if len(node_feat) < self.feature_dim:
                # Pad with zeros if we don't have enough history
                node_feat = np.pad(node_feat, (self.feature_dim - len(node_feat), 0))
            windows.append(node_feat)
        return torch.tensor(np.array(windows)).float()
