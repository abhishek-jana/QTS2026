import numpy as np
import pandas as pd
import pywt

class FractionalDifferencer:
    """
    Implements Fractional Differentiation (Approach B: Expanding Window)
    to preserve memory while ensuring stationarity.
    Formula: Delta^d x_t = sum_{k=0}^{\infty} omega_k x_{t-k}
    """
    def __init__(self, d: float, threshold: float = 1e-5):
        self.d = d
        self.threshold = threshold
        self.weights = None

    def _compute_weights(self, size: int):
        """Precomputes the weights for the expanding window."""
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (self.d - k + 1) / k
            w.append(w_k)
        return np.array(w[::-1]).reshape(-1, 1)

    def transform(self, series: pd.Series) -> pd.Series:
        """Applies the fractional differentiation to the series."""
        df = series.to_frame('val')
        # We use an expanding window approach
        # Note: For very large series, this can be optimized with a fixed window
        # but here we follow the "Correctness/Long-term" mandate.
        res = []
        for i in range(len(df)):
            # Weights for the current window size
            weights = self._compute_weights(i + 1)
            # Apply weights to the historical slice
            val = np.dot(df.iloc[:i+1].values.T, weights)[0,0]
            res.append(val)
        
        return pd.Series(res, index=series.index)

class WaveletFeatureGenerator:
    """
    Generates Market Spectrograms using Continuous Wavelet Transform (CWT)
    with Morlet wavelets on Logarithmic/Dyadic scales.
    """
    def __init__(self, scales: np.ndarray = None, wavelet: str = 'cmor1.5-1.0'):
        if scales is None:
            # Dyadic scales: 2^1 to 2^8 (capturing monthly/quarterly horizons)
            self.scales = 2 ** np.arange(1, 9)
        else:
            self.scales = scales
        self.wavelet = wavelet

    def generate(self, series: pd.Series) -> np.ndarray:
        """
        Returns a spectrogram of shape (n_scales, n_timesteps).
        """
        # Using PyWavelets for CWT
        # pywt.cwt returns [coefficients, frequencies]
        coefficients, _ = pywt.cwt(series.values, self.scales, self.wavelet)
        return np.abs(coefficients)

# Example usage for the notebook
if __name__ == "__main__":
    # Simulated long-term price series
    np.random.seed(42)
    t = np.linspace(0, 1, 500)
    price = pd.Series(np.cumsum(np.random.normal(0, 1, 500)) + 100)
    
    # 1. Fractional Differentiation
    fd = FractionalDifferencer(d=0.4)
    price_stationary = fd.transform(price)
    
    # 2. Wavelet Spectrogram
    wfg = WaveletFeatureGenerator()
    spectrogram = wfg.generate(price_stationary)
    
    print(f"Original shape: {price.shape}")
    print(f"Stationary series shape: {price_stationary.shape}")
    print(f"Spectrogram shape: {spectrogram.shape}")
