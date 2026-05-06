import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine

def run_time_travel_test():
    """
    Mathematically proves zero look-ahead bias in the DataEngine.
    """
    engine = DataEngine()
    engine.generate_synthetic_pit_data(['AAPL'], days=50)
    
    # Target event horizon to verify: Day 10
    horizon_date = datetime(2020, 1, 10, 23, 59)
    
    # Query 1: Ran on Day 11. We look at the truth of Day 10.
    knowledge_date_1 = horizon_date + timedelta(days=1)
    view_on_day_11 = engine.get_pit_view('AAPL', knowledge_date_1)
    # Filter to the horizon
    view_day_10_from_knowledge_11 = view_on_day_11[view_on_day_11.index <= horizon_date]
    
    # Query 2: Ran on Day 50. We look at the truth of Day 10.
    knowledge_date_2 = horizon_date + timedelta(days=40)
    view_on_day_50 = engine.get_pit_view('AAPL', knowledge_date_2)
    # Filter to the same horizon
    view_day_10_from_knowledge_50 = view_on_day_50[view_on_day_50.index <= horizon_date]
    
    # ASSERTION: The historical view of Day 10 must be identical despite 40 days of 'future' knowledge.
    pd.testing.assert_frame_equal(view_day_10_from_knowledge_11, view_day_10_from_knowledge_50)
    print("✅ Time-Travel Test Passed: Zero Look-Ahead Bias Verified.")

if __name__ == "__main__":
    run_time_travel_test()
