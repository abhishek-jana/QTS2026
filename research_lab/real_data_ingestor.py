import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from qts_core.logger import logger

class InstitutionalIngestor:
    """
    Production-grade data ingestor for Alpaca Market Data V2.
    Optimized for batch symbol requests and paginated historical retrieval.
    """
    def __init__(self, data_engine: DataEngine, config: dict = None):
        self.engine = data_engine
        self.config = config or {}
        self.provider = self.config.get('data_engine', {}).get('provider', 'ALPACA').upper()
        
        # Strict Credential Mapping
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = "https://data.alpaca.markets/v2/stocks/bars"

        if not self.api_key or not self.api_secret:
            logger.error("❌ ALPACA CREDENTIALS NOT FOUND in .env (Required: ALPACA_API_KEY, ALPACA_SECRET_KEY)")
        else:
            logger.info(f"InstitutionalIngestor: Initialized for {self.provider}")

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        try:
            existing = [t[0] for t in self.engine.conn.execute("SELECT DISTINCT ticker FROM market_data").fetchall()]
            missing = [t for t in tickers if t not in existing]
            if not missing:
                logger.info("✅ ALL TICKERS PRESENT: Skipping ingestion.")
                return
        except Exception: missing = tickers

        logger.info(f"📡 Alpaca Ingestion: Fetching {len(missing)} tickers...")
        headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
        all_data = []

        # Batch symbols in chunks of 100 to stay within URI limits
        for i in range(0, len(missing), 100):
            chunk = missing[i : i + 100]
            page_token = None
            
            while True:
                params = {
                    "symbols": ",".join(chunk), "timeframe": "1Day",
                    "start": start_date, "end": end_date,
                    "adjustment": "raw", "feed": "iex", "limit": 10000
                }
                if page_token: params["page_token"] = page_token
                
                resp = requests.get(self.base_url, headers=headers, params=params)
                if resp.status_code != 200:
                    logger.error(f"Alpaca API Error: {resp.status_code} - {resp.text}")
                    break
                
                res_json = resp.json()
                bars = res_json.get('bars', {})
                if not bars: break

                for ticker, ticker_bars in bars.items():
                    for bar in ticker_bars:
                        all_data.append({
                            'ticker': ticker, 'event_time': pd.to_datetime(bar['t']),
                            'knowledge_time': pd.to_datetime(bar['t']) + timedelta(hours=16),
                            'open': float(bar['o']), 'high': float(bar['h']),
                            'low': float(bar['l']), 'close': float(bar['c']),
                            'volume': int(bar['v']), 'is_correction': False
                        })
                
                page_token = res_json.get('next_page_token')
                if not page_token: break

        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            logger.success(f"✅ INGESTION COMPLETE: {len(all_data)} records stored in DuckDB.")
