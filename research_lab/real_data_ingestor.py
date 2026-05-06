import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from qts_core.logger import logger

class InstitutionalIngestor:
    """
    Ingests historical data from Polygon.io, Tiingo, or yfinance (fallback)
    into the PIT DataEngine.
    """
    def __init__(self, data_engine: DataEngine):
        self.engine = data_engine
        self.provider = os.getenv("DATA_PROVIDER", "YFINANCE").upper()
        
        if self.provider == "POLYGON":
            from polygon import RESTClient
            self.client = RESTClient(os.getenv("POLYGON_API_KEY"))
            logger.info("InstitutionalIngestor: Connected to Polygon.io")
        elif self.provider == "TIINGO":
            self.api_key = os.getenv("TIINGO_API_KEY")
            self.headers = {'Content-Type': 'application/json'}
            logger.info("InstitutionalIngestor: Connected to Tiingo")
        else:
            logger.info("InstitutionalIngestor: Using yfinance provider.")
            self.provider = "YFINANCE"

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        try:
            count = self.engine.conn.execute("SELECT count(*) FROM market_data").fetchone()[0]
            if count > 0:
                logger.info(f"CACHE HIT: Found {count} records in DuckDB. Skipping download.")
                return
        except Exception: pass

        logger.info(f"📡 FETCHING {self.provider} DATA: {len(tickers)} tickers...")
        all_data = []
        
        if self.provider == "YFINANCE":
            df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
            for ticker in tickers:
                ticker_data = df[ticker] if len(tickers) > 1 else df
                for event_time, row in ticker_data.iterrows():
                    if pd.isna(row['Close']): continue
                    all_data.append({
                        'ticker': ticker, 'event_time': event_time, 
                        'knowledge_time': event_time + timedelta(hours=16),
                        'open': float(row['Open']), 'high': float(row['High']), 
                        'low': float(row['Low']), 'close': float(row['Close']),
                        'volume': int(row['Volume']), 'is_correction': False
                    })
        
        elif self.provider == "POLYGON":
            for ticker in tickers:
                try:
                    # SENIOR FIX: Robust method discovery for different Polygon SDK versions
                    bars = []
                    if hasattr(self.client, 'list_aggs'):
                        bars = list(self.client.list_aggs(ticker, 1, "day", start_date, end_date, adjusted=False))
                    elif hasattr(self.client, 'get_aggs'):
                        bars = self.client.get_aggs(ticker, 1, "day", start_date, end_date, adjusted=False)
                    elif hasattr(self.client, 'stocks_equities_aggregates'):
                        resp = self.client.stocks_equities_aggregates(ticker, 1, "day", start_date, end_date, adjusted=False)
                        bars = resp.results if hasattr(resp, 'results') else []
                    
                    for bar in bars:
                        # Extract timestamp (some versions use .timestamp, others 't' or 'timestamp')
                        ts = bar.timestamp if hasattr(bar, 'timestamp') else getattr(bar, 't', 0)
                        event_time = pd.to_datetime(ts, unit='ms')
                        all_data.append({
                            'ticker': ticker, 'event_time': event_time, 
                            'knowledge_time': event_time + timedelta(hours=16),
                            'open': float(getattr(bar, 'open', getattr(bar, 'o', 0))), 
                            'high': float(getattr(bar, 'high', getattr(bar, 'h', 0))), 
                            'low': float(getattr(bar, 'low', getattr(bar, 'l', 0))), 
                            'close': float(getattr(bar, 'close', getattr(bar, 'c', 0))),
                            'volume': int(getattr(bar, 'volume', getattr(bar, 'v', 0))), 
                            'is_correction': False
                        })
                except Exception as e: logger.error(f"Polygon error for {ticker}: {e}")

        elif self.provider == "TIINGO":
            import requests
            for ticker in tickers:
                try:
                    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={self.api_key}"
                    resp = requests.get(url, headers=self.headers).json()
                    for bar in resp:
                        event_time = pd.to_datetime(bar['date'])
                        all_data.append({
                            'ticker': ticker, 'event_time': event_time, 
                            'knowledge_time': event_time + timedelta(hours=16),
                            'open': float(bar['open']), 'high': float(bar['high']), 
                            'low': float(bar['low']), 'close': float(bar['close']),
                            'volume': int(bar['volume']), 'is_correction': False
                        })
                except Exception as e: logger.error(f"Tiingo error for {ticker}: {e}")
        
        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            logger.success(f"INGESTION COMPLETE: {len(all_data)} records loaded into DuckDB.")
