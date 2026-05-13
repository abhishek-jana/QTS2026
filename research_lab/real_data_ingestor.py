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
        # Professional Range Analysis: Check what we already have in DuckDB
        missing_tails = {} # ticker -> adjusted_start
        missing_heads = {} # ticker -> adjusted_end
        
        target_start = pd.to_datetime(start_date).tz_localize('UTC') if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date).tz_localize('UTC') if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)
        if end_date == "now": target_end = pd.Timestamp.now(tz='UTC')

        logger.info("🔍 Analyzing local database for data gaps...")
        for ticker in tickers:
            res = self.engine.conn.execute(f"SELECT MIN(event_time), MAX(event_time) FROM market_data WHERE ticker = '{ticker}'").fetchone()
            if not res or res[0] is None:
                missing_tails[ticker] = start_date
            else:
                db_min = pd.to_datetime(res[0]).tz_localize('UTC') if pd.to_datetime(res[0]).tzinfo is None else pd.to_datetime(res[0])
                db_max = pd.to_datetime(res[1]).tz_localize('UTC') if pd.to_datetime(res[1]).tzinfo is None else pd.to_datetime(res[1])
                
                # Check for tail gap (new data since last run)
                if db_max < (target_end - timedelta(hours=1)):
                    missing_tails[ticker] = (db_max + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
                
                # Check for head gap (user requested older data than we have)
                if db_min > (target_start + timedelta(hours=1)):
                    missing_heads[ticker] = (db_min - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

        if not missing_tails and not missing_heads:
            logger.info("✅ DATABASE SYNCED: All requested data is already present.")
            return

        if self.provider == 'ALPACA':
            # Alpaca is better for bulk: We'll take the earliest missing start and fetch to now
            if missing_tails:
                earliest_start = min(missing_tails.values())
                self._ingest_alpaca(list(missing_tails.keys()), earliest_start, end_date)
            if missing_heads:
                # This is rare (user reaching back), fetch specific head gaps
                for t, head_end in missing_heads.items():
                    self._ingest_alpaca([t], start_date, head_end)
                    
        elif self.provider == 'TIINGO':
            # Tiingo is ticker-by-ticker, so we can be surgical
            all_missing = set(missing_tails.keys()) | set(missing_heads.keys())
            self._ingest_tiingo_surgical(all_missing, missing_tails, missing_heads, start_date, end_date)

    def _ingest_tiingo_surgical(self, tickers, tails, heads, global_start, global_end):
        import time as time_module
        logger.info(f"📡 Tiingo Surgical: Fetching {len(tickers)} tickers with adjusted ranges...")
        timeframe = self.config.get('data_engine', {}).get('timeframe', '15Min')
        resample_freq = timeframe.lower().replace('min', 'min').replace('day', 'day').replace('hour', 'hour')
        
        total_inserted = 0
        pbar = tqdm(tickers, desc="📊 Surgical Ingestion", unit="ticker")
        
        for ticker in pbar:
            # 1. Fetch Head (if requested older data)
            if ticker in heads:
                self._fetch_tiingo_range(ticker, global_start, heads[ticker], resample_freq)
            
            # 2. Fetch Tail (new data)
            if ticker in tails:
                self._fetch_tiingo_range(ticker, tails[ticker], global_end, resample_freq)
            
            time_module.sleep(0.5) # Pace
        pbar.close()

    def _fetch_tiingo_range(self, ticker, start, end, freq):
        import time as time_module
        url = f"https://api.tiingo.com/iex/{ticker}/prices"
        
        if end == "now":
            end = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
            
        params = {"startDate": start, "endDate": end, "resampleFreq": freq, "token": self.tiingo_api_key}
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params)
                if resp.status_code == 200:
                    bars = resp.json()
                    if bars:
                        df = pd.DataFrame([{
                            'ticker': ticker, 'event_time': pd.to_datetime(b['date']),
                            'knowledge_time': pd.to_datetime(b['date']),
                            'open': float(b['open']), 'high': float(b['high']),
                            'low': float(b['low']), 'close': float(b['close']),
                            'volume': int(b.get('volume', 0)), 'is_correction': False
                        } for b in bars])
                        self.engine.insert_dataframe(df)
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

    def _ingest_alpaca(self, tickers: list, start_date: str, end_date: str):
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        
        missing = tickers
        if not missing:
            logger.info("✅ ALL TICKERS IN RANGE: Skipping ingestion.")
            return

        logger.info(f"📡 Alpaca Ingestion: Fetching {len(missing)} tickers from {start_date}...")
        
        headers = {"APCA-API-KEY-ID": self.alpaca_api_key, "APCA-API-SECRET-KEY": self.alpaca_api_secret}
        # Force timeframe to match config (e.g. 15Min for Alpaca)
        timeframe = self.config.get('data_engine', {}).get('timeframe', '15Min')
        all_data = []
        total_inserted = 0

        pbar = tqdm(total=len(missing), desc="📊 Ingesting Alpaca Data", unit="ticker")

        # Convert dates to strict RFC3339 format for Alpaca
        def to_rfc3339(date_str):
            if date_str == "now":
                return pd.Timestamp.now(tz='UTC').isoformat()
            dt = pd.to_datetime(date_str)
            if dt.tzinfo is None: dt = dt.tz_localize('UTC')
            return dt.isoformat()
            
        alpaca_start = to_rfc3339(start_date)
        alpaca_end = to_rfc3339(end_date)

        for chunk_idx in range(0, len(missing), 100):
            chunk = missing[chunk_idx : chunk_idx + 100]
            page_token = None
            
            while True:
                params = {
                    "symbols": ",".join(chunk), "timeframe": timeframe,
                    "start": alpaca_start, "end": alpaca_end,
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
                        # Alpaca returns UTC. Convert to naive US/Eastern so it matches DuckDB storage format.
                        evt_est = evt_time.tz_convert('US/Eastern').tz_localize(None)
                        
                        # Market hours filter: 9:30 AM to 4:00 PM EST inclusive
                        if not (time(9, 30) <= evt_est.time() <= time(16, 0)): continue

                        k_time = evt_est if timeframe != '1Day' else evt_est.replace(hour=16, minute=0)
                            
                        all_data.append({
                            'ticker': ticker, 'event_time': evt_est, 'knowledge_time': k_time,
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
        
        if end_date == "now":
            end_date = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
            
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
                    'volume': int(bar.get('volume', 0)), 'is_correction': False
                })
            
            if df_bars:
                self.engine.insert_dataframe(pd.DataFrame(df_bars))
                total_inserted += len(df_bars)
            
            # Pacing
            time_module.sleep(1.0)
                
        pbar.close()
        logger.success(f"✅ TIINGO INGESTION COMPLETE: {total_inserted} records stored.")
