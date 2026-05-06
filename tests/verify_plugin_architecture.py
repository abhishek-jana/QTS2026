import sys
import os
sys.path.append(os.path.abspath('.'))

from datetime import datetime, timedelta
import torch
import numpy as np
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_plugins import ModalityPlugin

class DummySentimentPlugin(ModalityPlugin):
    """Example of a new modality plugin."""
    def transform(self, ticker, view, universe_views):
        # Return dummy sentiment (random)
        return np.random.rand(len(view), 1)

    @property
    def name(self):
        return "x_sentiment"

def test_plugin_architecture():
    universe = AlphaUniverse()
    tickers = ['AAPL', 'MSFT', 'SPY']
    universe.engine.generate_synthetic_pit_data(tickers, days=200)

    # 1. Test Default Plugins (Sequential + Spatial)
    as_of_date = datetime(2020, 1, 1) + timedelta(days=180)
    dataset = universe.get_aligned_dataset(
        tickers=tickers, 
        as_of_date=as_of_date, 
        lookback=63
    )
    
    print(f"Default Dataset Size: {len(dataset)}")
    print(f"x_seq shape: {dataset.x_seq.shape}")
    print(f"x_spatial shape: {dataset.x_spatial.shape}")
    
    assert dataset.x_seq is not None
    assert dataset.x_spatial is not None
    assert 'x_seq' in dataset[0]
    assert 'x_spatial' in dataset[0]

    # 2. Test Adding a New Plugin
    universe.add_plugin(DummySentimentPlugin())
    dataset_with_sent = universe.get_aligned_dataset(
        tickers=tickers, 
        as_of_date=as_of_date, 
        lookback=63
    )
    
    print(f"Extended Dataset modalities: {dataset_with_sent.modalities.keys()}")
    print(f"x_sentiment shape: {dataset_with_sent.modalities['x_sentiment'].shape}")
    
    assert 'x_sentiment' in dataset_with_sent.modalities
    assert 'x_sentiment' in dataset_with_sent[0]

    print("Plugin architecture verification SUCCESSFUL!")

if __name__ == "__main__":
    test_plugin_architecture()
