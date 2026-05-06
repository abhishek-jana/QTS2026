import pytest
import pandas as pd
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from alpha_factory.wfo_engine import WFOEngine

def test_wfo_knowledge_isolation():
    """
    TDD: Verify WFO Engine never sees data with knowledge_time > current_step.
    """
    engine = DataEngine()
    # Data at T=10, but only known at T=15
    event_time = datetime(2020, 1, 10)
    knowledge_time = datetime(2020, 1, 15)
    
    engine.registry = pd.DataFrame([{
        'ticker': 'AAPL',
        'event_time': event_time,
        'knowledge_time': knowledge_time,
        'close': 100.0
    }])
    
    wfo = WFOEngine(engine, {'input_dim': 10})
    
    # Step at T=12: Should NOT see the record
    current_date = datetime(2020, 1, 12)
    train_view = engine.registry[engine.registry['knowledge_time'] <= current_date]
    assert len(train_view) == 0
    
    # Step at T=16: SHOULD see the record
    current_date_2 = datetime(2020, 1, 16)
    train_view_2 = engine.registry[engine.registry['knowledge_time'] <= current_date_2]
    assert len(train_view_2) == 1
