import os
import pandas as pd
import requests
import time as time_module
from datetime import datetime, timedelta, time
from research_lab.data_engine import DataEngine
from qts_core.logger import logger
from tqdm import tqdm

class InstitutionalIngestor:
    """
    Production-grade data ingestor for Alpaca and Tiingo Market Data.
    Optimized for batch symbol requests and paginated historical retrieval.
    """
    def __init__(self, data_engine: DataEngine, config: dict = None):
        self.engine = data_engine
        self.config = config or {}
        self.provider = self.config.get('data_engine', {}).get('provider', 'ALPACA').upper()
        
        # Alpaca Credentials
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        
        # Tiingo Credentials
        self.tiingo_api_key = os.getenv("TIINGO_API_KEY")

        if self.provider == 'ALPACA':
            if not self.alpaca_api_key or not self.alpaca_api_secret:
                logger.error("❌ ALPACA CREDENTIALS NOT FOUND in .env")
            else:
                logger.info(f"InstitutionalIngestor: Initialized for {self.provider}")
        elif self.provider == 'TIINGO':
            if not self.tiingo_api_key:
                logger.error("❌ TIINGO CREDENTIALS NOT FOUND in .env")
            else:
                logger.info(f"InstitutionalIngestor: Initialized for {self.provider}")

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        if self.provider == 'ALPACA':
            self._ingest_alpaca(tickers, start_date, end_date)
        elif self.provider == 'TIINGO':
            self._ingest_tiingo(tickers, start_date, end_date)

    def _ingest_alpaca(self, tickers: list, start_date: str, end_date: str):
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        try:
            requested_start = pd.to_datetime(start_date)
            missing = []
            for ticker in tickers:
                res = self.engine.conn.execute(f"SELECT MIN(event_time) FROM market_data WHERE ticker = '{ticker}'").fetchone()
                if not res or res[0] is None or res[0] > requested_start:
                    missing.append(ticker)
            
            if not missing:
                logger.info("✅ ALL TICKERS IN RANGE: Skipping ingestion.")
                return
        except Exception as e: 
            logger.warning(f"Ingestion check failed ({e}), defaulting to full fetch.")
            missing = tickers

        logger.info(f"📡 Alpaca Ingestion: Fetching {len(missing)} tickers from {start_date}...")
        
        headers = {"APCA-API-KEY-ID": self.alpaca_api_key, "APCA-API-SECRET-KEY": self.alpaca_api_secret}
        timeframe = self.config.get('data_engine', {}).get('timeframe', '1Day')
        all_data = []
        total_inserted = 0

        pbar = tqdm(total=len(missing), desc="📊 Ingesting Alpaca Data", unit="ticker")

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
                
                resp = requests.get(base_url, headers=headers, params=params)
                if resp.status_code != 200:
                    logger.error(f"Alpaca API Error: {resp.status_code} - {resp.text}")
                    break
                
                res_json = resp.json()
                bars = res_json.get('bars', {})
                if not bars: break

                for ticker, ticker_bars in bars.items():
                    pbar.set_postfix_str(f"Current: {ticker}")
                    pbar.update(1)

                    for bar in ticker_bars:
                        evt_time = pd.to_datetime(bar['t'])
                        evt_est = evt_time.tz_convert('US/Eastern') if evt_time.tzinfo else evt_time.tz_localize('UTC').tz_convert('US/Eastern')
                        if not (time(9, 30) <= evt_est.time() < time(16, 0)): continue

                        k_time = evt_time if timeframe != '1Day' else evt_time.replace(hour=16, minute=0)
                            
                        all_data.append({
                            'ticker': ticker, 'event_time': evt_time, 'knowledge_time': k_time,
                            'open': float(bar['o']), 'high': float(bar['h']), 'low': float(bar['l']), 
                            'close': float(bar['c']), 'volume': int(bar['v']), 'is_correction': False
                        })
                        
                        if len(all_data) >= 50000:
                            self.engine.insert_dataframe(pd.DataFrame(all_data))
                            total_inserted += len(all_data)
                            all_data = []
                
                page_token = res_json.get('next_page_token')
                if not page_token: break
        
        pbar.close()
        if all_data:
            self.engine.insert_dataframe(pd.DataFrame(all_data))
            total_inserted += len(all_data)
        logger.success(f"✅ ALPACA INGESTION COMPLETE: {total_inserted} records stored.")

    def _ingest_tiingo(self, tickers: list, start_date: str, end_date: str):
        logger.info(f"📡 Tiingo Ingestion: Fetching {len(tickers)} tickers from {start_date}...")
        timeframe = self.config.get('data_engine', {}).get('timeframe', '15Min')
        
        resample_freq = timeframe.lower().replace('min', 'min').replace('day', 'day').replace('hour', 'hour')
        
        total_inserted = 0
        pbar = tqdm(tickers, desc="📊 Ingesting Tiingo Data", unit="ticker")
        
        for ticker in pbar:
            pbar.set_postfix_str(f"Current: {ticker}")
            url = f"https://api.tiingo.com/iex/{ticker}/prices"
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": resample_freq,
                "columns": "open,high,low,close,volume",
                "token": self.tiingo_api_key
            }
            
            # RETRY LOGIC WITH EXPONENTIAL BACKOFF
            max_retries = 5
            base_delay = 2.0
            bars = None
            
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, params=params)
                    if resp.status_code == 200:
                        bars = resp.json()
                        break 
                    elif resp.status_code == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited on {ticker}. Retrying in {delay:.1f}s...")
                        time_module.sleep(delay)
                    else:
                        logger.error(f"Tiingo Error ({ticker}): {resp.status_code}")
                        break
                except Exception as e:
                    logger.error(f"Request Exception ({ticker}): {e}")
                    break
            else:
                logger.error(f"Failed to fetch {ticker} after {max_retries} retries.")

            if not bars: 
                time_module.sleep(1.0)
                continue
            
            df_bars = []
            for bar in bars:
                evt_time = pd.to_datetime(bar['date'])
                evt_est = evt_time.tz_convert('US/Eastern') if evt_time.tzinfo else evt_time.tz_localize('UTC').tz_convert('US/Eastern')
                if not (time(9, 30) <= evt_est.time() < time(16, 0)): continue
                
                df_bars.append({
                    'ticker': ticker, 'event_time': evt_time,
                    'knowledge_time': evt_time if timeframe != '1Day' else evt_time.replace(hour=16, minute=0),
                    'open': float(bar['open']), 'high': float(bar['high']),
                    'low': float(bar['low']), 'close': float(bar['close']),
                    'volume': int(bar['volume']), 'is_correction': False
                })
            
            if df_bars:
                self.engine.insert_dataframe(pd.DataFrame(df_bars))
                total_inserted += len(df_bars)
            
            # Pacing
            time_module.sleep(1.0)
                
        pbar.close()
        logger.success(f"✅ TIINGO INGESTION COMPLETE: {total_inserted} records stored.")
