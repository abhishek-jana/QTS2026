import pandas as pd
import numpy as np
import duckdb
import yaml
from datetime import datetime, timedelta
from typing import List, Protocol, Optional

class IDataProvider(Protocol):
    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame: ...
    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame: ...

class DataEngine:
    def __init__(self, storage_path: str = "data/uqts_bitemporal.ddb", config_path: str = "config.yaml", read_only: bool = False):
        self.storage_path = storage_path
        # SENIOR FIX: Support read_only mode to allow concurrent access from UI and Worker
        self.conn = duckdb.connect(self.storage_path, read_only=read_only)
        self.config_path = config_path
        self.features = ['close']
        self._load_config()
        if not read_only:
            self._init_db()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.features = config.get('data_engine', {}).get('features', ['close'])
        except Exception: pass

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
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tk ON market_data (ticker, knowledge_time)")

    def insert_dataframe(self, df: pd.DataFrame):
        if df.empty: return
        self.conn.register("df", df)
        self.conn.execute("INSERT INTO market_data SELECT * FROM df")
        self.conn.unregister("df")

    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame:
        if not tickers: return pd.DataFrame()
        ticker_list = "', '".join(tickers)
        
        # SENIOR FIX: Always include standard OHLCV columns + any extra requested features
        # This prevents 'KeyError: open' in visualization layers.
        standard_cols = ['ticker', 'event_time', 'knowledge_time', 'open', 'high', 'low', 'close', 'volume', 'is_correction']
        feature_cols = [f for f in self.features if f not in standard_cols]
        all_cols = standard_cols + feature_cols
        col_str = ", ".join(all_cols)
        
        query = f"""
            WITH ranked_data AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker, event_time ORDER BY knowledge_time DESC) as rn
                FROM market_data
                WHERE ticker IN ('{ticker_list}') AND knowledge_time <= '{as_of.isoformat()}'
            )
            SELECT {col_str} FROM ranked_data WHERE rn = 1 ORDER BY event_time ASC
        """
        pit_view = self.conn.execute(query).fetchdf()
        if pit_view.empty: return pit_view
        return pit_view.set_index('event_time')

    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame:
        return self.get_batch_pit_view([ticker], as_of)
