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
        elif self.provider == "TIINGO":
            import requests
            self.api_key = os.getenv("TIINGO_API_KEY")
            self.headers = {'Content-Type': 'application/json'}
        elif self.provider == "YFINANCE":
            logger.info("InstitutionalIngestor: Using yfinance provider.")
        else:
            logger.warning(f"InstitutionalIngestor: Unknown provider {self.provider}, falling back to YFINANCE.")
            self.provider = "YFINANCE"

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        # 0. CHECK CACHE: Skip if data already exists for these tickers
        try:
            # Simple check: If any data exists, we skip for now. 
            # In a production system, we'd check date ranges per ticker.
            count = self.engine.conn.execute("SELECT count(*) FROM market_data").fetchone()[0]
            if count > 0:
                logger.info(f"CACHE HIT: Found {count} records in DuckDB. Skipping download.")
                return
        except Exception as e:
            logger.warning(f"Cache check failed (possibly empty DB): {e}")

        logger.info(f"FETCHING {self.provider} DATA: {tickers}...")
        all_data = []
        
        if self.provider == "YFINANCE":
            try:
                # yfinance returns data indexed by Date
                df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
                if df.empty:
                    logger.warning("yfinance returned no data.")
                    return

                for ticker in tickers:
                    ticker_data = df[ticker] if len(tickers) > 1 else df
                    for event_time, row in ticker_data.iterrows():
                        if pd.isna(row['Close']): continue
                        all_data.append({
                            'ticker': ticker, 
                            'event_time': event_time, 
                            'knowledge_time': event_time + timedelta(hours=16),
                            'open': float(row['Open']), 
                            'high': float(row['High']), 
                            'low': float(row['Low']), 
                            'close': float(row['Close']),
                            'volume': int(row['Volume']), 
                            'is_correction': False
                        })
            except Exception as e:
                logger.error(f"yfinance ingestion error: {e}")
        
        else:
            for ticker in tickers:
                try:
                    if self.provider == "POLYGON":
                        bars = self.client.get_aggs(ticker, 1, "day", start_date, end_date, adjusted=False)
                        for bar in bars:
                            all_data.append(self._format_polygon(ticker, bar))
                    elif self.provider == "TIINGO":
                        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={self.api_key}"
                        import requests
                        resp = requests.get(url, headers=self.headers).json()
                        for bar in resp:
                            all_data.append(self._format_tiingo(ticker, bar))
                except Exception as e:
                    logger.error(f"Error fetching {ticker} from {self.provider}: {e}")
        
        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            logger.info(f"INGESTION COMPLETE: {len(all_data)} records loaded.")

    def _format_polygon(self, ticker, bar):
        event_time = pd.to_datetime(bar.timestamp, unit='ms')
        return {'ticker': ticker, 'event_time': event_time, 'knowledge_time': event_time + timedelta(hours=16),
                'open': float(bar.open), 'high': float(bar.high), 'low': float(bar.low), 'close': float(bar.close),
                'volume': int(bar.volume), 'is_correction': False}

    def _format_tiingo(self, ticker, bar):
        event_time = pd.to_datetime(bar['date'])
        return {'ticker': ticker, 'event_time': event_time, 'knowledge_time': event_time + timedelta(hours=16),
                'open': float(bar['open']), 'high': float(bar['high']), 'low': float(bar['low']), 'close': float(bar['close']),
                'volume': int(bar['volume']), 'is_correction': False}
