import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from research_lab.alpha_universe import AlphaUniverse

def test_alpha_universe_alignment():
    # Setup Data Engine with synthetic data
    engine = DataEngine()
    tickers = ['AAPL', 'MSFT', 'SPY']
    engine.generate_synthetic_pit_data(tickers, days=100)
    
    universe = AlphaUniverse(data_engine=engine)
    
    # Parameters
    as_of_date = datetime(2020, 1, 1) + timedelta(days=90)
    horizon = 5
    d_param = 0.4
    scales = 2 ** np.arange(1, 4) # 3 scales for speed
    
    # Get aligned dataset
    dataset = universe.get_aligned_dataset(
        tickers=tickers,
        as_of_date=as_of_date,
        horizon=horizon,
        d_param=d_param,
        scales=scales
    )
    
    # Assertions
    assert not dataset.empty
    assert 'event_time' in dataset.columns
    assert 'ticker' in dataset.columns
    assert 'label' in dataset.columns
    
    # Check if scales are present
    for i in range(len(scales)):
        assert f'scale_{i}' in dataset.columns
        
    # Check ticker coverage
    # SPY is used as market proxy and might be dropped from labels if handled that way
    present_tickers = dataset['ticker'].unique()
    assert 'AAPL' in present_tickers
    assert 'MSFT' in present_tickers
    
    # Check bi-temporal alignment: no labels for the last 'horizon' days
    max_event_time = dataset['event_time'].max()
    # Data was generated for 100 days starting 2020-01-01
    # as_of_date is day 90.
    # The last event_time should be around day 90 - horizon.
    expected_max_date = datetime(2020, 1, 1) + timedelta(days=90 - horizon - 1)
    assert max_event_time <= expected_max_date
    
    # Check that labels are Z-scored (mean ~ 0)
    # Group by event_time and check mean
    daily_means = dataset.groupby('event_time')['label'].mean()
    assert np.allclose(daily_means, 0, atol=1e-7)

if __name__ == "__main__":
    test_alpha_universe_alignment()
