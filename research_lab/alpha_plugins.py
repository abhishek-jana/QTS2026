from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch
from research_lab.alpha_core import FractionalDifferencer, WaveletFeatureGenerator

class ModalityPlugin(ABC):
    """
    Interface for AlphaUniverse data modalities.
    Enables plug-and-play addition of new feature streams (e.g., Sentiment, GNN).
    """
    
    @abstractmethod
    def transform(self, ticker: str, view: pd.DataFrame, universe_views: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Transforms raw ticker data into a feature time-series.
        
        Args:
            ticker: The symbol being processed.
            view: The Point-in-Time (PIT) view for this ticker.
            universe_views: PIT views for all tickers in the universe.
            
        Returns:
            np.ndarray of shape (T, *FeatureDims) aligned with 'view' indices.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the modality (e.g., 'x_seq', 'x_spatial')."""
        pass

    def prepare_universe(self, universe_views: Dict[str, pd.DataFrame]):
        """
        Optional: Global pre-computation hook for the entire universe 
        (e.g., building a correlation graph for GNNs).
        """
        pass

    def post_process_batch(self, batch: np.ndarray) -> torch.Tensor:
        """
        Optional: Final transformation on the stacked batch (N, lookback, *FeatureDims).
        Default implementation converts to a float tensor.
        """
        return torch.tensor(batch).float()

class SequentialModality(ModalityPlugin):
    """
    Sequential Stream: Fractionally Differenced price series.
    Optimized for LSTM-based encoders.
    """
    def __init__(self, d: float = 0.4):
        self.fd = FractionalDifferencer(d=d)

    def transform(self, ticker: str, view: pd.DataFrame, universe_views: Dict[str, pd.DataFrame]) -> np.ndarray:
        stationary = self.fd.transform(view['close']).values
        return stationary.reshape(-1, 1) # Shape: (T, 1)

    @property
    def name(self) -> str:
        return "x_seq"

class SpatialModality(ModalityPlugin):
    """
    Spatial Stream: 2D Wavelet Spectrogram.
    Optimized for Vision Transformer (ViT) encoders.
    """
    def __init__(self, d: float = 0.4, scales: np.ndarray = None):
        self.fd = FractionalDifferencer(d=d)
        self.wfg = WaveletFeatureGenerator(scales=scales)

    def transform(self, ticker: str, view: pd.DataFrame, universe_views: Dict[str, pd.DataFrame]) -> np.ndarray:
        stationary = self.fd.transform(view['close'])
        spectrogram = self.wfg.generate(stationary) # Shape: (Scales, T)
        return spectrogram.T # Transpose to (T, Scales) to match time alignment

    @property
    def name(self) -> str:
        return "x_spatial"

    def post_process_batch(self, batch: np.ndarray) -> torch.Tensor:
        # batch: (N, lookback, Scales)
        # ViT expects (N, Channels=1, Height=Scales, Width=lookback)
        tensor = torch.tensor(batch).float()
        return tensor.transpose(1, 2).unsqueeze(1)
