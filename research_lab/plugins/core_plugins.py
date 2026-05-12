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
        
        # SENIOR FIX: Per-Scale Z-Score Normalization
        # Wavelet coefficients vary wildly by scale. ViT requires stable standard deviations.
        means = np.mean(spectrogram, axis=1, keepdims=True)
        stds = np.std(spectrogram, axis=1, keepdims=True) + 1e-8
        spectrogram_norm = (spectrogram - means) / stds

        windows = [spectrogram_norm[:, t-lookback:t] for t in range(lookback, len(stationary) + 1)]
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

@register_modality("x_calendar")
class CalendarPlugin(ModalityPlugin):
    """Encodes calendar effects (day of week, month, etc.) as temporal features."""
    def __init__(self, **kwargs): self._name = "x_calendar"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        # Extract features from index (DatetimeIndex expected)
        times = pit_view.index
        dow = times.weekday.values / 6.0 # Normalize 0-1
        month = (times.month.values - 1) / 11.0
        hour = times.hour.values / 23.0
        minute = times.minute.values / 59.0
        
        features = np.stack([dow, month, hour, minute], axis=1)
        windows = [features[t-lookback:t] for t in range(lookback, len(features) + 1)]
        return torch.tensor(np.array(windows)).float()

@register_modality("x_static")
class StaticMetadataPlugin(ModalityPlugin):
    """Provides static stock metadata (Sector, Market Cap Decile) as covariates."""
    def __init__(self, **kwargs): 
        self._name = "x_static"
        # Dummy mapping for demo; in production, this should be a DB lookup
        self.sector_map = {
            'AAPL': 0, 'MSFT': 0, 'NVDA': 0, 'GOOGL': 0, 'AMZN': 0, # Tech/Comm
            'META': 0, 'TSLA': 1, 'LLY': 2, 'UNH': 2, 'JPM': 3,    # Cons/Health/Fin
            'V': 3, 'MA': 3, 'AVGO': 0, 'HD': 4, 'PG': 4,          # Fin/Cons
            'COST': 4, 'JNJ': 2, 'ABBV': 2, 'MRK': 2, 'BAC': 3,
            'CRM': 0, 'ORCL': 0, 'ADBE': 0, 'AMD': 0, 'PEP': 4,
            'KO': 4, 'TMO': 2, 'WMT': 4, 'MCD': 4, 'CSCO': 0,
            'NFLX': 0, 'ABT': 2, 'DHR': 2, 'WFC': 3, 'ACN': 0,
            'QCOM': 0, 'LIN': 5, 'GE': 5, 'PM': 4, 'TXN': 0,
            'INTU': 0, 'AMGN': 2, 'VZ': 0, 'AMAT': 0, 'UNP': 5,
            'LOW': 4, 'BX': 3, 'GS': 3, 'ISRG': 2, 'HON': 5,
            'MS': 3, 'CVS': 2, 'COP': 6, 'IBM': 0, 'BA': 5,
            'SPGI': 3, 'CAT': 5, 'LMT': 5, 'RTX': 5, 'SPY': 7
        }
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        ticker = pit_view['ticker'].iloc[0]
        sector = self.sector_map.get(ticker, 8) # 8 = Unknown
        # For now, just a single static categorical feature per ticker
        # We repeat it across the windows (though TFT only needs it once)
        val = torch.tensor([float(sector)])
        n_windows = len(pit_view) - lookback + 1
        return val.repeat(n_windows, 1)

@register_modality("x_momentum")
class MomentumPlugin(ModalityPlugin):
    """Provides raw trend visibility (no fractional differencing)."""
    def __init__(self, **kwargs): self._name = "x_momentum"
    @property
    def name(self) -> str: return self._name
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        close = pit_view['close'].values
        ret_10 = np.zeros_like(close)
        ret_20 = np.zeros_like(close)
        ret_60 = np.zeros_like(close)
        
        # SENIOR FIX: Safe division to prevent RuntimeWarning on zero-padded data
        for i in range(10, len(close)): 
            ret_10[i] = (close[i] / close[i-10]) - 1.0 if abs(close[i-10]) > 1e-6 else 0.0
        for i in range(20, len(close)): 
            ret_20[i] = (close[i] / close[i-20]) - 1.0 if abs(close[i-20]) > 1e-6 else 0.0
        for i in range(60, len(close)): 
            ret_60[i] = (close[i] / close[i-60]) - 1.0 if abs(close[i-60]) > 1e-6 else 0.0
        
        # Combine into features (T, 3)
        features = np.stack([ret_10, ret_20, ret_60], axis=1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) # Double-check safety
        
        windows = [features[t-lookback:t] for t in range(lookback, len(features) + 1)]
        return torch.tensor(np.array(windows)).float()
