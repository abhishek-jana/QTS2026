import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from research_lab.alpha_universe import AlphaUniverse

def test_alpha_universe_multimodal_dataset():
    """
    TDD: Verify AlphaUniverse produces correct MultiModalDataset shapes.
    """
    engine = DataEngine()
    tickers = ['AAPL', 'MSFT', 'SPY']
    # Need enough data for lookback (63) + labels
    engine.generate_synthetic_pit_data(tickers, days=150)
    
    universe = AlphaUniverse(engine)
    lookback = 63
    as_of = datetime(2020, 1, 1) + timedelta(days=150)
    
    dataset = universe.get_aligned_dataset(
        tickers=['AAPL', 'MSFT', 'SPY'], 
        as_of_date=as_of, 
        lookback=lookback
    )
    
    assert dataset is not None
    assert len(dataset) > 0
    
    # Check shapes
    # x_seq: (N, 63, 1)
    # x_spatial: (N, 1, 8, 63)
    # y: (N)
    assert dataset.x_seq.dim() == 3
    assert dataset.x_seq.shape[1] == lookback
    assert dataset.x_seq.shape[2] == 1
    
    assert dataset.x_spatial.dim() == 4
    assert dataset.x_spatial.shape[1] == 1
    assert dataset.x_spatial.shape[2] == 8
    assert dataset.x_spatial.shape[3] == lookback
    
    assert dataset.y.dim() == 1
    assert len(dataset.y) == len(dataset.x_seq)
    
    print(f"Dataset Size: {len(dataset)}")
