import pandas as pd
import numpy as np
import duckdb
import yaml
import os
from datetime import datetime, timedelta
from typing import List, Protocol, Optional, Dict
from qts_core.logger import logger

class IDataProvider(Protocol):
    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame: ...
    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame: ...

class DataEngineRegistry:
    """SENIOR SINGLETON: Manages process-wide DuckDB connections to prevent lock conflicts."""
    _connections: Dict[str, duckdb.DuckDBPyConnection] = {}
    
    @classmethod
    def get_connection(cls, path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
        key = f"{path}_{read_only}"
        if key not in cls._connections:
            # First, check if a connection with a DIFFERENT mode exists
            other_mode = not read_only
            other_key = f"{path}_{other_mode}"
            if other_key in cls._connections:
                logger.warning(f"⚠️ DataEngineRegistry: Switch detected ({other_mode} -> {read_only}). Re-opening {path}.")
                try:
                    cls._connections[other_key].close()
                    del cls._connections[other_key]
                except Exception: pass
            
            cls._connections[key] = duckdb.connect(path, read_only=read_only)
        return cls._connections[key]

class DataEngine:
    def __init__(self, storage_path: str = "data/uqts_bitemporal.ddb", config_path: str = "config.yaml", read_only: bool = False):
        self.storage_path = storage_path
        self.read_only = read_only
        self.config_path = config_path
        self.features = ['close']
        self._load_config()
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # SENIOR DEV FIX: Only init tables if we aren't in a read-only context
        if not read_only:
            try:
                self._init_db()
            except duckdb.IOException as e:
                if "Conflicting lock" in str(e):
                    logger.warning("⚠️ DataEngine: Write lock held by another process. Forcing READ-ONLY.")
                    self.read_only = True

    @property
    def conn(self):
        # Always use the registry to get the shared connection handle
        return DataEngineRegistry.get_connection(self.storage_path, self.read_only)

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.features = config.get('data_engine', {}).get('features', ['close'])
        except Exception: pass

    def _init_db(self):
        # We use a temporary connection for initialization to ensure we don't pollute the singleton
        temp_conn = duckdb.connect(self.storage_path, read_only=False)
        temp_conn.execute("""
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
        temp_conn.execute("CREATE INDEX IF NOT EXISTS idx_tk ON market_data (ticker, knowledge_time)")
        temp_conn.close()

    def close(self):
        # Note: Singleton handles stay open for process lifetime unless explicitly evicted
        pass

    def insert_dataframe(self, df: pd.DataFrame):
        if df.empty or self.read_only: return
        self.conn.register("df", df)
        self.conn.execute("INSERT INTO market_data SELECT * FROM df")
        self.conn.unregister("df")

    def get_batch_pit_view(self, tickers: List[str], as_of: datetime, start_time: Optional[datetime] = None) -> pd.DataFrame:
        if not tickers: return pd.DataFrame()
        ticker_list = "', '".join(tickers)
        
        standard_cols = ['ticker', 'event_time', 'knowledge_time', 'open', 'high', 'low', 'close', 'volume', 'is_correction']
        feature_cols = [f for f in self.features if f not in standard_cols]
        all_cols = standard_cols + feature_cols
        col_str = ", ".join(all_cols)
        
        time_filter = f"AND knowledge_time <= '{as_of.isoformat()}'"
        if start_time:
            time_filter += f" AND event_time >= '{start_time.isoformat()}'"
            
        query = f"""
            WITH ranked_data AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker, event_time ORDER BY knowledge_time DESC) as rn
                FROM market_data
                WHERE ticker IN ('{ticker_list}') {time_filter}
            )
            SELECT {col_str} FROM ranked_data WHERE rn = 1 ORDER BY event_time ASC
        """
        pit_view = self.conn.execute(query).df()
        if pit_view.empty: return pit_view
        return pit_view.set_index('event_time')

    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame:
        return self.get_batch_pit_view([ticker], as_of)
