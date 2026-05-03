import pytest
import torch
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from research_lab.alpha_universe import AlphaUniverse
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin

def test_unified_lab_snapshot():
    """
    TDD: Verify Unified Lab Snapshot provides multi-modal alignment.
    """
    engine = DataEngine()
    tickers = ['AAPL', 'MSFT', 'SPY']
    engine.generate_synthetic_pit_data(tickers, days=150)
    
    # 1. Register Plugins
    plugins = [SequentialPlugin(d_param=0.4), SpatialPlugin()]
    
    # 2. Orchestrate
    universe = AlphaUniverse(engine, plugins=plugins)
    as_of = datetime(2020, 1, 1) + timedelta(days=120)
    
    batch = universe.snapshot(as_of=as_of, tickers=tickers, lookback=63)
    
    assert batch is not None
    # Check Modalities
    assert "x_seq" in batch.data
    assert "x_spatial" in batch.data
    
    # Check Alignment
    N = len(batch.labels)
    assert batch.data['x_seq'].shape[0] == N
    assert batch.data['x_spatial'].shape[0] == N
    
    print(f"Unified Batch Size: {N}")
    print(f"Modality Seq Shape: {batch['x_seq'].shape}")
    print(f"Modality Spatial Shape: {batch['x_spatial'].shape}")

def test_unified_lab_walk_forward():
    """
    TDD: Verify Walk-Forward automation steps through history.
    """
    engine = DataEngine()
    tickers = ['AAPL', 'SPY']
    engine.generate_synthetic_pit_data(tickers, days=200)
    
    plugins = [SequentialPlugin()]
    universe = AlphaUniverse(engine, plugins=plugins)
    
    start = datetime(2020, 1, 1) + timedelta(days=100)
    end = datetime(2020, 1, 1) + timedelta(days=150)
    
    # stride=21 (Monthly retrains)
    results = universe.walk_forward(universe=tickers, start_date=start, end_date=end, stride=21)
    
    # Should have ~3 steps (Day 100, 121, 142)
    assert len(results) >= 2
    assert 'batch' in results[0]
    print(f"Walk-forward steps completed: {len(results)}")
