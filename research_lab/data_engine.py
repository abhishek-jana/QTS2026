import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataEngine:
    """
    Bi-temporal Data Engine for Point-in-Time (PIT) consistency.
    Ensures that "Knowledge Time" (when the system learned the info) 
    is strictly separated from "Event Time" (when the market event happened).
    """
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self.registry = None # In-memory placeholder for QuestDB/DuckDB

    def generate_synthetic_pit_data(self, tickers: list, days: int = 1000):
        """
        Generates synthetic OHLCV data with simulated knowledge delays.
        Example: A correction for price at T happened at T+2.
        """
        np.random.seed(42)
        data = []
        start_date = datetime(2020, 1, 1)
        
        for ticker in tickers:
            price = 100.0
            for d in range(days):
                event_time = start_date + timedelta(days=d)
                # Market moves
                price *= (1 + np.random.normal(0, 0.01))
                
                # Knowledge Time: Usually Event Time + small delay (e.g., end of day)
                # But sometimes we have late-arriving data or corrections.
                knowledge_time = event_time + timedelta(hours=16) # Market close
                
                # Standard record
                data.append({
                    'ticker': ticker,
                    'event_time': event_time,
                    'knowledge_time': knowledge_time,
                    'open': price * 0.99,
                    'high': price * 1.01,
                    'low': price * 0.98,
                    'close': price,
                    'volume': np.random.randint(1000, 100000)
                })
                
                # SIMULATE CORRECTION: 1% chance a price is revised 2 days later
                if np.random.rand() < 0.01:
                    revision_knowledge_time = event_time + timedelta(days=2, hours=9)
                    data.append({
                        'ticker': ticker,
                        'event_time': event_time,
                        'knowledge_time': revision_knowledge_time,
                        'open': price * 0.99,
                        'high': price * 1.01,
                        'low': price * 0.98,
                        'close': price * 1.02, # The correction
                        'volume': np.random.randint(1000, 100000),
                        'is_correction': True
                    })
        
        self.registry = pd.DataFrame(data)
        return self.registry

    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame:
        """
        Returns the "Best Known Truth" for a ticker as of a specific knowledge time.
        This is the core PIT interface.
        """
        if self.registry is None:
            raise ValueError("Data registry is empty. Ingest data first.")
            
        # 1. Filter by knowledge_time (Knowledge isolation)
        visible_data = self.registry[
            (self.registry['ticker'] == ticker) & 
            (self.registry['knowledge_time'] <= as_of)
        ]
        
        # 2. Get the latest record for each event_time (Handling corrections)
        # We sort by knowledge_time descending to get the most recent 'truth'
        pit_view = visible_data.sort_values('knowledge_time', ascending=False).drop_duplicates('event_time')
        
        return pit_view.sort_values('event_time').set_index('event_time')

if __name__ == "__main__":
    engine = DataEngine()
    engine.generate_synthetic_pit_data(['AAPL', 'MSFT'])
    
    # Test PIT consistency
    test_date = datetime(2020, 1, 10)
    knowledge_date_1 = datetime(2020, 1, 10, 23, 59)
    knowledge_date_2 = datetime(2020, 1, 15, 23, 59)
    
    view_1 = engine.get_pit_view('AAPL', knowledge_date_1)
    view_2 = engine.get_pit_view('AAPL', knowledge_date_2)
    
    print(f"Data known as of {knowledge_date_1}: {len(view_1)} rows")
    print(f"Data known as of {knowledge_date_2}: {len(view_2)} rows")
    
    # Check if a 1/10 price correction (arriving on 1/12) is invisible to view_1
    # but visible to view_2.
