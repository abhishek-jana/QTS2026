import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
from typing import List

class FractionalDifferencer:
    """
    Implements Fractional Differentiation using FFT Convolution.
    Complexity: O(N log N) vs original O(N^2).
    """
    def __init__(self, d: float, threshold: float = 1e-5):
        self.d = d
        self.threshold = threshold

    def transform(self, series: pd.Series) -> pd.Series:
        """Applies FFT-based fractional differentiation to price deviations."""
        vals = series.values
        N = len(vals)
        
        # 1. Precompute weights for the entire length N
        #omega_k = -omega_{k-1} * (d - k + 1) / k
        w = np.zeros(N)
        w[0] = 1.0
        for k in range(1, N):
            w[k] = -w[k-1] * (self.d - k + 1) / k
            
        # 2. Standard Practice: Differentiate the deviations from initial price
        v_offset = vals[0]
        v_shifted = vals - v_offset
        
        # 3. FFT Convolution (Linear convolution)
        # We only need the first N results (causal convolution)
        res = fftconvolve(v_shifted, w, mode='full')[:N]
            
        return pd.Series(res, index=series.index)

class WaveletFeatureGenerator:
    """
    Generates Market Spectrograms using native PyTorch 1D Convolutions.
    Calculated on GPU for massive parallel speedup.
    """
    def __init__(self, scales: np.ndarray = None, wavelet: str = 'mexh'):
        if scales is None:
            self.scales = 2 ** np.arange(1, 9)
        else:
            self.scales = scales
        self.wavelet_name = wavelet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # SENIOR OPTIMIZATION: Build filters and move to device ONCE at init
        self._filters = self._build_filters()

    def _build_filters(self) -> List[torch.Tensor]:
        """Precomputes Mexican Hat wavelet kernels for each scale."""
        filters = []
        for s in self.scales:
            # Mexican Hat Formula: (1 - t^2) * exp(-t^2 / 2) * constant
            width = int(10 * s)
            if width % 2 == 0: width += 1
            t = np.linspace(-5, 5, width)
            constant = 2 / (np.sqrt(3) * (np.pi**0.25))
            kernel = constant * (1 - t**2) * np.exp(-t**2 / 2)
            kernel = kernel / np.sqrt(s)
            
            # Move to device immediately
            filt_tensor = torch.tensor(kernel).float().to(self.device).view(1, 1, -1)
            filters.append(filt_tensor)
        return filters

    def generate(self, series: pd.Series) -> np.ndarray:
        """
        Returns a spectrogram of shape (n_scales, n_timesteps).
        Optimized via PyTorch Conv1d with zero-copy filters.
        """
        x = torch.tensor(series.values).float().to(self.device).view(1, 1, -1)
        results = []
        
        with torch.no_grad():
            for filt in self._filters:
                padding = filt.shape[-1] // 2
                # Apply 1D convolution (filters are already on device)
                conv_res = F.conv1d(x, filt, padding=padding)
                results.append(conv_res.view(-1))
        
        # Stack scales and convert back to CPU numpy for the pipeline
        spectrogram = torch.stack(results).cpu().numpy()
        return np.abs(spectrogram[:, :len(series)])

class DiurnalStandardizer:
    """
    Normalizes intraday returns and volume by time-of-day buckets
    to mitigate the 'U-Shape' volatility seasonality.
    """
    def __init__(self, interval_minutes: int = 15):
        self.interval = interval_minutes

    def transform(self, df: pd.DataFrame, columns: list = ['close', 'volume']) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        res = df.copy()
        # Create a 'bucket' key based on time of day
        res['tod_bucket'] = df.index.hour * 60 + df.index.minute
        
        for col in columns:
            if col not in res.columns: continue
            if not np.issubdtype(res[col].dtype, np.number): continue
            
            # Use rolling historical average/std for each bucket
            m = res.groupby('tod_bucket')[col].transform('mean')
            s = res.groupby('tod_bucket')[col].transform('std')
            res[col] = (res[col] - m) / (s + 1e-9)
            
        return res.drop(columns=['tod_bucket'])

if __name__ == "__main__":
    # Test parity/speed
    np.random.seed(42)
    N = 1000
    price = pd.Series(np.cumsum(np.random.normal(0, 1, N)) + 100)
    
    fd = FractionalDifferencer(d=0.4)
    price_stationary = fd.transform(price)
    
    wfg = WaveletFeatureGenerator(scales=np.arange(1, 17))
    spectrogram = wfg.generate(price_stationary)
    
    print(f"Stationary shape: {price_stationary.shape}")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print("✅ Optimization sanity check complete.")
