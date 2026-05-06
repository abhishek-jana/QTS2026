import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
from typing import List, Protocol

class IDataProvider(Protocol):
    """
    Interface for PIT-consistent data providers.
    """
    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame: ...
    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame: ...

class DataEngine:
    """
    Bi-temporal Data Engine for Point-in-Time (PIT) consistency using DuckDB.
    """
    def __init__(self, storage_path: str = "data/uqts_bitemporal.ddb"):
        self.storage_path = storage_path
        self.conn = duckdb.connect(self.storage_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker VARCHAR,
                event_time TIMESTAMP,
                knowledge_time TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                is_correction BOOLEAN DEFAULT FALSE
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_knowledge 
            ON market_data (ticker, knowledge_time);
        """)

    def insert_dataframe(self, df: pd.DataFrame):
        """Inserts a pandas DataFrame into the market_data table."""
        if df.empty:
            return
        # Explicitly register the dataframe so DuckDB can see it
        self.conn.register("df", df)
        self.conn.execute("INSERT INTO market_data SELECT * FROM df")
        self.conn.unregister("df")

    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame:
        return self.get_batch_pit_view([ticker], as_of)

    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame:
        """
        Fetches PIT-consistent views for multiple tickers in a single optimized query.
        Crucial for scaling to 10k+ tickers.
        """
        if not tickers:
            return pd.DataFrame()
            
        ticker_list = "', '".join(tickers)
        query = f"""
            WITH ranked_data AS (
                SELECT *,
                       ROW_NUMBER() OVER(
                           PARTITION BY ticker, event_time 
                           ORDER BY knowledge_time DESC
                       ) as rn
                FROM market_data
                WHERE ticker IN ('{ticker_list}') 
                  AND knowledge_time <= '{as_of.isoformat()}'
            )
            SELECT ticker, event_time, knowledge_time, open, high, low, close, volume, is_correction
            FROM ranked_data
            WHERE rn = 1
            ORDER BY event_time ASC
        """
        
        pit_view = self.conn.execute(query).fetchdf()
        
        if pit_view.empty:
            return pit_view
            
        return pit_view.set_index('event_time')

    def generate_synthetic_pit_data(self, tickers: list, days: int = 1000):
        """
        Generates synthetic OHLCV data with simulated knowledge delays.
        """
        np.random.seed(42)
        data = []
        start_date = datetime(2020, 1, 1)

        for ticker in tickers:
            price = 100.0
            for d in range(days):
                event_time = start_date + timedelta(days=d)
                price *= (1 + np.random.normal(0, 0.01))
                knowledge_time = event_time + timedelta(hours=16)

                data.append({
                    'ticker': ticker,
                    'event_time': event_time,
                    'knowledge_time': knowledge_time,
                    'open': price * 0.99,
                    'high': price * 1.01,
                    'low': price * 0.98,
                    'close': price,
                    'volume': np.random.randint(1000, 100000),
                    'is_correction': False
                })

                if np.random.rand() < 0.01:
                    revision_knowledge_time = event_time + timedelta(days=2, hours=9)
                    data.append({
                        'ticker': ticker,
                        'event_time': event_time,
                        'knowledge_time': revision_knowledge_time,
                        'open': price * 0.99,
                        'high': price * 1.01,
                        'low': price * 0.98,
                        'close': price * 1.02,
                        'volume': np.random.randint(1000, 100000),
                        'is_correction': True
                    })

        df = pd.DataFrame(data)
        self.insert_dataframe(df)
        return df
