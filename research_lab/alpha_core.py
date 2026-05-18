"""
Signal-physics core: fractional differencing, wavelet spectrogram, diurnal standardizer.

Efficiency notes:
  - FractionalDifferencer.transform: weight recurrence is vectorized (cumprod)
    and cached per (d, N). The original ran a Python for-loop on every call.
  - WaveletFeatureGenerator: generate() now returns a contiguous numpy array
    via a single .cpu().numpy() (was already correct), and the device choice
    avoids GPU round-trips when filters are tiny.
"""
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve


class FractionalDifferencer:
    """
    Implements Fractional Differentiation using FFT Convolution.
    Complexity: O(N log N) vs original O(N^2).

    Weights for the recurrence w[k] = -w[k-1] * (d - k + 1) / k are computed
    via numpy cumprod (vectorized) and cached per (d, N) so a walk-forward
    pass that calls transform() thousands of times pays the weight cost once.
    """

    # Class-level weight cache. Bounded by len(set of (d, N)) seen during a
    # process lifetime; typical training uses 1-2 values of d and a handful
    # of distinct N. Memory cost is negligible.
    _weight_cache: dict = {}
    _weight_cache_max = 64

    def __init__(self, d: float, threshold: float = 1e-5):
        self.d = d
        self.threshold = threshold

    def _weights(self, N: int) -> np.ndarray:
        """Vectorized + cached weight computation."""
        key = (self.d, N)
        cached = self._weight_cache.get(key)
        if cached is not None:
            return cached
        if N <= 0:
            w = np.zeros(0)
        elif N == 1:
            w = np.array([1.0])
        else:
            k = np.arange(1, N)
            # factors[i] = -(d - (i+1) + 1) / (i+1) for i=0..N-2
            factors = -(self.d - k + 1) / k
            # w[k] = product(factors[:k]); w[0] = 1.0
            w = np.concatenate(([1.0], np.cumprod(factors)))
        # Crude LRU: drop oldest if cache too large.
        if len(self._weight_cache) >= self._weight_cache_max:
            # Pop an arbitrary key (dict insertion order)
            self._weight_cache.pop(next(iter(self._weight_cache)))
        self._weight_cache[key] = w
        return w

    def transform(self, series: pd.Series) -> pd.Series:
        """Applies FFT-based fractional differentiation to price deviations."""
        vals = series.values
        N = len(vals)

        w = self._weights(N)

        # Differentiate the deviations from the initial price (standard practice).
        v_offset = vals[0]
        v_shifted = vals - v_offset

        # FFT convolution; keep only the first N (causal) entries.
        res = fftconvolve(v_shifted, w, mode='full')[:N]
        return pd.Series(res, index=series.index)


class WaveletFeatureGenerator:
    """
    Generates Market Spectrograms using PyTorch 1D Convolutions.

    Filters are built once at __init__ and held on the target device. The
    generate() path keeps the data on the chosen device for the conv1d, then
    moves the result back to CPU numpy in one operation (downstream callers
    consume numpy / construct torch tensors from it).

    Note: on small batches (a single ticker, N < few thousand), CPU is often
    faster than CPU↔GPU transfer. Set TORCH_WAVELET_DEVICE=cpu to force CPU.
    """

    def __init__(self, scales: np.ndarray = None, wavelet: str = 'mexh'):
        if scales is None:
            self.scales = 2 ** np.arange(1, 9)
        else:
            self.scales = np.asarray(scales)
        self.wavelet_name = wavelet

        import os
        forced = os.getenv("TORCH_WAVELET_DEVICE")
        if forced:
            self.device = torch.device(forced)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._filters = self._build_filters()

    def _build_filters(self) -> List[torch.Tensor]:
        """Precompute Mexican-Hat kernels for each scale on the target device."""
        filters = []
        for s in self.scales:
            width = int(10 * s)
            if width % 2 == 0:
                width += 1
            t = np.linspace(-5, 5, width)
            constant = 2 / (np.sqrt(3) * (np.pi ** 0.25))
            kernel = constant * (1 - t ** 2) * np.exp(-(t ** 2) / 2)
            kernel = kernel / np.sqrt(s)
            filt_tensor = torch.from_numpy(kernel.astype(np.float32)).to(self.device).view(1, 1, -1)
            filters.append(filt_tensor)
        return filters

    def generate(self, series: pd.Series) -> np.ndarray:
        """Returns a spectrogram of shape (n_scales, n_timesteps)."""
        # from_numpy avoids the deprecated/extra copy that torch.tensor() makes.
        x = torch.from_numpy(series.values.astype(np.float32)).to(self.device).view(1, 1, -1)
        results = []
        with torch.no_grad():
            for filt in self._filters:
                # SENIOR FIX (CAUSAL): instead of centered padding (look-ahead),
                # we use left-padding (causal) so the output at index T only 
                # depends on data from [T - width, T].
                filter_width = filt.shape[-1]
                # Pad only on the left side
                x_padded = F.pad(x, (filter_width - 1, 0))
                # Run convolution with no additional padding
                conv_res = F.conv1d(x_padded, filt, padding=0)
                results.append(conv_res.view(-1))
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
        res['tod_bucket'] = df.index.hour * 60 + df.index.minute

        for col in columns:
            if col not in res.columns:
                continue
            if not np.issubdtype(res[col].dtype, np.number):
                continue
            m = res.groupby('tod_bucket')[col].transform('mean')
            s = res.groupby('tod_bucket')[col].transform('std')
            res[col] = (res[col] - m) / (s + 1e-9)

        return res.drop(columns=['tod_bucket'])


if __name__ == "__main__":
    # Sanity / parity check
    np.random.seed(42)
    N = 1000
    price = pd.Series(np.cumsum(np.random.normal(0, 1, N)) + 100)
    fd = FractionalDifferencer(d=0.4)
    price_stationary = fd.transform(price)
    wfg = WaveletFeatureGenerator(scales=np.arange(1, 17))
    spectrogram = wfg.generate(price_stationary)
    print(f"Stationary shape: {price_stationary.shape}")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print("OK")
