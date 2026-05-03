import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ModalityPlugin(ABC):
    """
    Protocol for adding new data modalities (features) to AlphaUniverse.
    """
    @abstractmethod
    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        """
        Takes a PIT view and returns a feature tensor of shape (N, ...).
        N must match the number of aligned event_times in the view.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class SequentialPlugin(ModalityPlugin):
    """LSTM Stream: 1D Fractionally Differenced Returns."""
    def __init__(self, d_param: float = 0.4):
        from research_lab.alpha_core import FractionalDifferencer
        self.fd = FractionalDifferencer(d=d_param)
        self._name = "x_seq"

    @property
    def name(self) -> str:
        return self._name

    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        windows = []
        for t in range(lookback, len(stationary) + 1):
            windows.append(stationary[t-lookback:t].reshape(-1, 1))
        return torch.tensor(np.array(windows)).float()

class SpatialPlugin(ModalityPlugin):
    """ViT Stream: 2D Wavelet Spectrograms with High Density Scales."""
    def __init__(self, scales: np.ndarray = None):
        from research_lab.alpha_core import WaveletFeatureGenerator
        from research_lab.alpha_core import FractionalDifferencer
        self.fd = FractionalDifferencer(d=0.4) 
        # Increase density: 32 scales to provide a "Signal Energy" view
        self.scales = scales if scales is not None else np.geomspace(2, 256, 32)
        self.wfg = WaveletFeatureGenerator(scales=self.scales)
        self._name = "x_spatial"

    @property
    def name(self) -> str:
        return self._name

    def transform(self, pit_view: pd.DataFrame, lookback: int) -> torch.Tensor:
        stationary = self.fd.transform(pit_view['close']).values
        spectrogram = self.wfg.generate(pd.Series(stationary))
        windows = []
        for t in range(lookback, len(stationary) + 1):
            windows.append(spectrogram[:, t-lookback:t])
        # Add channel dimension: (N, 1, Scales, Lookback)
        return torch.tensor(np.array(windows)).unsqueeze(1).float()
