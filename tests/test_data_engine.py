import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine

def test_pit_isolation():
    """
    RED: Test that data known in the future is invisible at an earlier knowledge time.
    """
    engine = DataEngine()
    ticker = "AAPL"
    event_time = datetime(2020, 1, 1)
    
    # Truth 1: Original price known on day 1
    knowledge_time_1 = event_time + timedelta(hours=16)
    record_1 = {
        'ticker': ticker,
        'event_time': event_time,
        'knowledge_time': knowledge_time_1,
        'close': 100.0
    }
    
    # Truth 2: Correction known on day 3
    knowledge_time_2 = event_time + timedelta(days=2)
    record_2 = {
        'ticker': ticker,
        'event_time': event_time,
        'knowledge_time': knowledge_time_2,
        'close': 105.0 # The revision
    }
    
    # Ingest data
    engine.registry = pd.DataFrame([record_1, record_2])
    
    # Query AS OF Day 2 (Knowledge time 1 is visible, Knowledge time 2 is NOT)
    as_of_day_2 = event_time + timedelta(days=1)
    view = engine.get_pit_view(ticker, as_of_day_2)
    
    assert len(view) == 1
    assert view.iloc[0]['close'] == 100.0
    
    # Query AS OF Day 3 (Knowledge time 2 is now visible and should override)
    as_of_day_3 = event_time + timedelta(days=3)
    view_updated = engine.get_pit_view(ticker, as_of_day_3)
    
    assert len(view_updated) == 1
    assert view_updated.iloc[0]['close'] == 105.0

def test_feature_alignment():
    """
    RED: Test that the full pipeline (DataEngine -> AlphaCore) maintains temporal alignment.
    """
    from research_lab.alpha_core import FractionalDifferencer, WaveletFeatureGenerator
    
    engine = DataEngine()
    # Generate 100 days of synthetic data for a single ticker
    engine.generate_synthetic_pit_data(['AAPL'], days=100)
    
    # Fetch PIT view as of the end of the period
    as_of = datetime(2020, 1, 1) + timedelta(days=101)
    view = engine.get_pit_view('AAPL', as_of)
    
    # 1. Fractional Differentiation
    fd = FractionalDifferencer(d=0.4)
    stationary_series = fd.transform(view['close'])
    
    # 2. Wavelet Spectrogram
    wfg = WaveletFeatureGenerator()
    spectrogram = wfg.generate(stationary_series)
    
    # Assertions
    assert len(stationary_series) == len(view)
    assert (stationary_series.index == view.index).all()
    assert spectrogram.shape[1] == len(view)
    assert spectrogram.shape[0] == len(wfg.scales)

def test_residualized_labeling():
    """
    RED: Test that the AlphaLabeler produces idiosyncratic residuals.
    """
    from research_lab.alpha_labeler import AlphaLabeler
    
    # 1. Create highly correlated asset and market series
    np.random.seed(42)
    market_returns = np.random.normal(0.0005, 0.01, 100)
    # Asset = 1.5 * Market + Noise
    asset_returns = 1.5 * market_returns + np.random.normal(0, 0.002, 100)
    
    labeler = AlphaLabeler()
    # Residualize asset returns against market
    residuals = labeler.residualize(asset_returns, market_returns)
    
    # Assertions
    # Residuals should have ~zero correlation with the market proxy
    correlation = np.corrcoef(residuals, market_returns)[0, 1]
    assert abs(correlation) < 1e-10
    assert len(residuals) == len(asset_returns)

def test_cross_sectional_zscoring():
    """
    RED: Test that the AlphaLabeler correctly Z-scores labels across tickers.
    """
    from research_lab.alpha_labeler import AlphaLabeler
    labeler = AlphaLabeler()
    
    # Create mock residuals for 5 tickers
    residuals = pd.DataFrame({
        'AAPL': [0.01, 0.02, 0.03],
        'MSFT': [-0.01, 0.0, 0.01],
        'GOOG': [0.05, -0.05, 0.0],
        'AMZN': [0.1, 0.1, 0.1],
        'META': [-0.1, -0.1, -0.1]
    })
    
    z_scores = labeler.apply_z_score(residuals)
    
    # Assertions for each time step (row)
    for i in range(len(z_scores)):
        row = z_scores.iloc[i]
        assert abs(row.mean()) < 1e-10
        # Check std only if there is variation in the row
        if row.std() > 0:
            assert abs(row.std() - 1.0) < 1e-10

def test_multi_ticker_pipeline():
    """
    RED: Test the full labeling pipeline: Forward Returns -> Batch Residualization -> Z-Scoring.
    """
    from research_lab.alpha_labeler import AlphaLabeler
    labeler = AlphaLabeler()
    
    # 1. Mock forward returns for 3 tickers + 1 market proxy
    dates = pd.date_range('2020-01-01', periods=10)
    returns_df = pd.DataFrame({
        'AAPL': np.random.normal(0, 0.01, 10),
        'MSFT': np.random.normal(0, 0.01, 10),
        'GOOG': np.random.normal(0, 0.01, 10),
        'SPY': np.random.normal(0, 0.005, 10)
    }, index=dates)
    
    # 2. Residualize each ticker against SPY
    market_proxy = returns_df['SPY']
    asset_returns = returns_df.drop(columns=['SPY'])
    
    residuals = labeler.residualize_universe(asset_returns, market_proxy)
    
    # 3. Z-Score
    final_labels = labeler.apply_z_score(residuals)
    
    # Assertions
    assert final_labels.shape == asset_returns.shape
    assert abs(final_labels.iloc[0].mean()) < 1e-10
    # Verify residuals are indeed idiosyncratic to SPY
    for ticker in ['AAPL', 'MSFT', 'GOOG']:
        corr = np.corrcoef(residuals[ticker], market_proxy)[0, 1]
        assert abs(corr) < 1e-10
