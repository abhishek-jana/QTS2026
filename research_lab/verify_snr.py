import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from research_lab.alpha_universe import AlphaUniverse
from research_lab.plugins.core_plugins import SpatialPlugin
from datetime import datetime, timedelta

def verify_signal_to_noise():
    """
    Verifies the predictive power of Wavelet features using Mutual Information.
    Target: Next-day idiosyncratic returns.
    """
    print("🔍 INITIATING SNR VERIFICATION (Mutual Information Test)...")
    
    # 1. Setup Lab
    tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY']
    spatial_plugin = SpatialPlugin()
    universe = AlphaUniverse(plugins=[spatial_plugin])
    
    # Generate data
    universe.engine.generate_synthetic_pit_data(tickers, days=600)
    as_of = datetime(2020, 10, 1)
    
    # 2. Get Aligned Dataset (lookback=63, horizon=1 for next-day)
    batch = universe.snapshot(as_of=as_of, tickers=tickers, lookback=63, horizon=1)
    
    if batch is None:
        print("❌ Error: No aligned data for MI test.")
        return

    # 3. Prepare Features (Spectrogram flattened) and Targets
    # Features: x_spatial is (N, 1, Scales, T). We'll take the mean energy per scale.
    X = torch_mean_energy = batch.data['x_spatial'].mean(dim=3).squeeze().numpy()
    y = batch.labels.numpy()
    
    # 4. Calculate Mutual Information
    # I(X; Y) measures how much information X provides about Y.
    mi_scores = mutual_info_regression(X, y)
    
    # 5. Report Results
    scales = np.arange(1, 9)
    results = pd.DataFrame({
        'Scale': [f"2^{s}" for s in scales],
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    print("\n--- Mutual Information: Wavelet Scales vs. Next-Day Returns ---")
    print(results.to_string(index=False))
    
    avg_mi = np.mean(mi_scores)
    print(f"\nAverage Mutual Information: {avg_mi:.4f}")
    
    if avg_mi > 0.005: # Threshold for "non-zero" predictive signal in noisy markets
        print("✅ SNR VERIFIED: Wavelet features contain predictive signal.")
    else:
        print("⚠️ SNR WARNING: Low information gain detected. Physics tuning required.")

if __name__ == "__main__":
    verify_signal_to_noise()
