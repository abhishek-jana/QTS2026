import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from research_lab.data_engine import DataEngine
from research_lab.alpha_core import FractionalDifferencer, WaveletFeatureGenerator
from datetime import datetime, timedelta

def verify_physics():
    engine = DataEngine()
    tickers = ['AAPL', 'MSFT', 'GOOG']
    engine.generate_synthetic_pit_data(tickers, days=500)
    
    as_of = datetime(2020, 1, 1) + timedelta(days=500)
    fd = FractionalDifferencer(d=0.4)
    wfg = WaveletFeatureGenerator()
    
    results = []
    for t in tickers:
        view = engine.get_pit_view(t, as_of)
        price = view['close']
        
        # 1. ADF Test on Raw vs Stationary
        raw_adf = adfuller(price)[1]
        stat_series = fd.transform(price)
        stat_adf = adfuller(stat_series)[1]
        
        # 2. Spectrogram entropy/activation
        spec = wfg.generate(stat_series)
        mean_amp = np.mean(spec, axis=1)
        
        results.append({
            'ticker': t,
            'raw_p_val': raw_adf,
            'stat_p_val': stat_adf,
            'scale_activation': mean_amp
        })
        
    for res in results:
        print(f"--- {res['ticker']} ---")
        print(f"ADF p-val (Raw): {res['raw_p_val']:.4f}")
        print(f"ADF p-val (Stationary): {res['stat_p_val']:.4e}")
        print(f"Scale Activation (2^1 to 2^8): {res['scale_activation']}")
        print("")

if __name__ == "__main__":
    verify_physics()
