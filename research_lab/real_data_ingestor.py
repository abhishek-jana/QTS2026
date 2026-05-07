import os
import pandas as pd
import requests
from datetime import datetime, timedelta, time
from research_lab.data_engine import DataEngine
from qts_core.logger import logger
from tqdm import tqdm

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
        # Support both naming conventions for secrets
        self.api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        self.base_url = "https://data.alpaca.markets/v2/stocks/bars"

        if not self.api_key or not self.api_secret:
            logger.error("❌ ALPACA CREDENTIALS NOT FOUND in .env (Required: ALPACA_API_KEY, ALPACA_SECRET_KEY/API_SECRET)")
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
        
        # SENIOR DEV FIX: Rebuild headers from current attributes to support dynamic injection
        api_key = getattr(self, 'alpaca_api_key', self.api_key)
        api_secret = getattr(self, 'alpaca_api_secret', getattr(self, 'alpaca_secret_key', self.api_secret))
        
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        timeframe = self.config.get('data_engine', {}).get('timeframe', '1Day')
        all_data = []
        total_inserted = 0

        # SENIOR PATTERN: Use TQDM for clean, single-line progress tracking
        pbar = tqdm(total=len(missing), desc="📊 Ingesting Market Data", unit="ticker")

        # Batch symbols in chunks of 100 to stay within URI limits
        for chunk_idx in range(0, len(missing), 100):
            chunk = missing[chunk_idx : chunk_idx + 100]
            page_token = None
            
            while True:
                params = {
                    "symbols": ",".join(chunk), "timeframe": timeframe,
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
                    # Update progress bar
                    pbar.set_postfix_str(f"Current: {ticker}")
                    pbar.update(1)

                    for bar in ticker_bars:
                        evt_time = pd.to_datetime(bar['t'])
                        
                        # ML RESEARCH FIX: Filter for Regular Trading Hours (09:30 - 16:00 EST)
                        evt_est = evt_time.tz_convert('US/Eastern') if evt_time.tzinfo else evt_time.tz_localize('UTC').tz_convert('US/Eastern')
                        if not (time(9, 30) <= evt_est.time() < time(16, 0)):
                            continue

                        if timeframe == '1Day':
                            k_time = evt_time.replace(hour=16, minute=0)
                        else:
                            k_time = evt_time # Known immediately at bar end
                            
                        all_data.append({
                            'ticker': ticker, 'event_time': evt_time,
                            'knowledge_time': k_time,
                            'open': float(bar['o']), 'high': float(bar['h']),
                            'low': float(bar['l']), 'close': float(bar['c']),
                            'volume': int(bar['v']), 'is_correction': False
                        })
                        
                        # SENIOR FIX: Incremental insertion to prevent RAM exhaustion
                        if len(all_data) >= 50000:
                            self.engine.insert_dataframe(pd.DataFrame(all_data))
                            total_inserted += len(all_data)
                            all_data = [] # Clear memory
                
                page_token = res_json.get('next_page_token')
                if not page_token: break
        
        pbar.close()

        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            total_inserted += len(all_data)
            
        logger.success(f"✅ INGESTION COMPLETE: {total_inserted} records stored in DuckDB.")
