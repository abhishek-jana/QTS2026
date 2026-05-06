import os
import pandas as pd
import yfinance as yf
import requests
import yaml
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from qts_core.logger import logger

class InstitutionalIngestor:
    def __init__(self, data_engine: DataEngine, config: dict = None):
        self.engine = data_engine
        self.config = config or {}
        self.provider = self.config.get('data_engine', {}).get('provider', os.getenv("DATA_PROVIDER", "YFINANCE")).upper()
        
        # SENIOR DEV FIX: Support multiple naming conventions for Alpaca keys
        self.api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
        self.api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")

        if self.provider == "ALPACA":
            if not self.api_key or not self.api_secret:
                logger.error("❌ ALPACA CREDENTIALS INCOMPLETE")
                if not self.api_key: logger.error("  > Missing ALPACA_API_KEY")
                if not self.api_secret: logger.error("  > Missing ALPACA_SECRET_KEY (or ALPACA_API_SECRET)")
            else:
                logger.info(f"InstitutionalIngestor: Connected to Alpaca (Key: {self.api_key[:4]}...)")

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        try:
            existing_tickers = self.engine.conn.execute("SELECT DISTINCT ticker FROM market_data").fetchall()
            existing_tickers = [t[0] for t in existing_tickers]
            missing_tickers = [t for t in tickers if t not in existing_tickers]
            if not missing_tickers:
                logger.info(f"✅ CACHE HIT: All tickers exist in DuckDB.")
                return
        except Exception: missing_tickers = tickers

        logger.info(f"📡 FETCHING {self.provider} DATA: {len(missing_tickers)} tickers...")
        all_data = []
        
        if self.provider == "ALPACA":
            headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
            chunk = missing_tickers[:100]
            # Try all data endpoints
            combos = [
                ("https://data.alpaca.markets/v2/stocks/bars", "iex"),
                ("https://data.alpaca.markets/v2/stocks/bars", "sip"),
                ("https://paper-api.alpaca.markets/v2/stocks/bars", "iex")
            ]
            for url, feed in combos:
                params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start_date, "end": end_date, "adjustment": "raw", "feed": feed}
                resp = requests.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    logger.info(f"✅ SUCCESS: {url} [{feed}]")
                    data = resp.json().get('bars', {})
                    for ticker, bars in data.items():
                        for bar in bars:
                            all_data.append({'ticker': ticker, 'event_time': pd.to_datetime(bar['t']), 'knowledge_time': pd.to_datetime(bar['t']) + timedelta(hours=16),
                                           'open': float(bar['o']), 'high': float(bar['h']), 'low': float(bar['l']), 'close': float(bar['c']),
                                           'volume': int(bar['v']), 'is_correction': False})
                    break
                else:
                    logger.warning(f"❌ FAILED: {url} [{feed}] -> {resp.status_code}")
        
        elif self.provider == "TIINGO":
            tiingo_key = os.getenv("TIINGO_API_KEY")
            for ticker in missing_tickers:
                url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={tiingo_key}"
                resp = requests.get(url)
                if resp.status_code == 200:
                    for bar in resp.json():
                        all_data.append({'ticker': ticker, 'event_time': pd.to_datetime(bar['date']), 'knowledge_time': pd.to_datetime(bar['date']) + timedelta(hours=16),
                                       'open': float(bar['open']), 'high': float(bar['high']), 'low': float(bar['low']), 'close': float(bar['close']),
                                       'volume': int(bar['volume']), 'is_correction': False})
        
        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            logger.success(f"✅ INGESTION COMPLETE: {len(all_data)} records loaded.")
