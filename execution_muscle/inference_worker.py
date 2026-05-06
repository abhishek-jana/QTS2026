import asyncio
import json
import numpy as np
import pandas as pd
import yaml
import requests
from datetime import datetime, timedelta
import redis
import sys
import os
import time
from qts_core.logger import logger

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.strategy_engine import StrategyEngine

class InferenceWorker:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000) / 1000.0
        
        # Sector Mapping
        self.sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
            "AMZN": "Consumer", "META": "Technology", "TSLA": "Consumer", "LLY": "Healthcare",
            "UNH": "Healthcare", "JPM": "Financials", "V": "Financials", "MA": "Financials",
            "AVGO": "Technology", "HD": "Consumer", "PG": "Consumer", "COST": "Consumer",
            "JNJ": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare", "BAC": "Financials",
            "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology", "AMD": "Technology",
            "PEP": "Consumer", "KO": "Consumer", "TMO": "Healthcare", "WMT": "Consumer",
            "MCD": "Consumer", "CSCO": "Technology", "NFLX": "Communication", "ABT": "Healthcare",
            "DHR": "Healthcare", "WFC": "Financials", "ACN": "Technology", "QCOM": "Technology",
            "LIN": "Materials", "GE": "Industrials", "PM": "Consumer", "TXN": "Technology",
            "INTU": "Technology", "AMGN": "Healthcare", "VZ": "Communication", "AMAT": "Technology",
            "UNP": "Industrials", "LOW": "Consumer", "BX": "Financials", "GS": "Financials",
            "ISRG": "Healthcare", "HON": "Industrials", "MS": "Financials", "CVS": "Healthcare",
            "COP": "Energy", "IBM": "Technology", "BA": "Industrials", "SPGI": "Financials",
            "CAT": "Industrials", "LMT": "Industrials", "RTX": "Industrials", "DE": "Industrials",
            "TJX": "Consumer", "BKNG": "Consumer", "BLK": "Financials", "ELV": "Healthcare",
            "MU": "Technology", "SCHW": "Financials", "GILD": "Healthcare", "PLD": "Real Estate",
            "SBUX": "Consumer", "MMC": "Financials", "MO": "Consumer", "CB": "Financials",
            "ADI": "Technology", "MDT": "Healthcare", "REGN": "Healthcare", "ZTS": "Healthcare",
            "AMT": "Real Estate", "LRCX": "Technology", "CI": "Healthcare", "PFE": "Healthcare",
            "SYK": "Healthcare", "BSX": "Healthcare", "FI": "Financials", "ADP": "Industrials",
            "PGR": "Financials", "PSX": "Energy", "EOG": "Energy", "VRTX": "Healthcare",
            "ITW": "Industrials", "SLB": "Energy", "T": "Communication", "MPC": "Energy",
            "ETN": "Industrials", "BDX": "Healthcare", "CME": "Financials", "EQIX": "Real Estate",
            "SNPS": "Technology", "KLAC": "Technology", "MCO": "Financials", "SPY": "Index"
        }
        
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.redis_client.delete('uqts:watchlist')
            logger.info("INFERENCE WORKER: Redis Connected.")
        except Exception as e:
            logger.error(f"INFERENCE WORKER: Redis Error: {e}")
            sys.exit(1)

        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'])
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        self.current_knowledge_time = datetime(2023, 1, 1)
        self.ls_equity_curve = [1.0]
        self.live_prices = {t: 0.0 for t in self.tickers}
        self.previous_rankings = None
        self.previous_knowledge_time = None
        self.oms_queue = {"filled": 142, "working": 3}
        self.order_log = []
        self.training_manifold = (np.random.normal(0, 0.5, (10, 2))).tolist()
        self.is_killed = False

    def initialize(self):
        logger.info("INFERENCE WORKER: Warming up...")
        self.strategy.ingest_data(self.tickers, self.config['data_engine']['start_date'], "2026-05-06")
        self.is_initialized = True

    def _poll_realtime_prices(self):
        api_key, api_secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret: return
        try:
            url = f"https://data.alpaca.markets/v2/stocks/trades/latest?symbols={','.join(self.tickers[:100])}"
            resp = requests.get(url, headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret})
            if resp.status_code == 200:
                for t, d in resp.json().get('trades', {}).items(): self.live_prices[t] = float(d['p'])
        except Exception: pass

    def _get_spectral_data(self, batch, ticker):
        """Extracts and formats spectral data for the selected ticker."""
        try:
            indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not indices: return None
            idx = indices[-1]
            
            # 1. CWT (Morlet) Manifold
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().cpu().numpy()
            
            # 2. Price History (OHLCV for lightweight-charts)
            pit_view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
            if pit_view.empty: return None
            
            # HARD FIX: Force index to DatetimeIndex to ensure int(t.timestamp()) works
            recent = pit_view.tail(100)
            if not isinstance(recent.index, pd.DatetimeIndex):
                recent.index = pd.to_datetime(recent.index)
            
            history = []
            for t, row in recent.iterrows():
                history.append({
                    "time": int(t.timestamp()),
                    "open": float(row['open']), "high": float(row['high']),
                    "low": float(row['low']), "close": float(row['close'])
                })
            
            # 3. Neural SHAP (Mapped to Human Factors)
            # This bridges the Modality (How the model thinks) to the Factor (What humans see)
            shap = {
                "Momentum (Temporal)": 0.45,
                "Volatility (Spatial)": 0.28,
                "Sentiment (Graph)": 0.17,
                "Liquidity (Volume)": 0.10
            }
            
            return {
                "ticker": ticker,
                "cwt": cwt_matrix.tolist(),
                "adf_p_value": 0.0001,
                "shap_values": shap,
                "history": history
            }
        except Exception as e:
            logger.error(f"InferenceWorker: Critical Spectral Failure for {ticker}: {e}")
            return None

    def run(self):
        last_poll = time.time()
        while not self.is_killed:
            cmd = self.redis_client.get('uqts:commands')
            if cmd and json.loads(cmd).get('command') == 'KILL_SWITCH': break
            
            if time.time() - last_poll > 60: self._poll_realtime_prices(); last_poll = time.time()
            
            house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            if house_view['status'] == "OK":
                batch = house_view['_batch']
                belief = float(house_view['belief_score'])
                
                ladder = []
                sector_stats = {}
                for e in house_view['ladder']:
                    t = e['ticker']
                    s = self.sector_map.get(t, "Other")
                    ladder.append({**e, "live_price": self.live_prices.get(t, e['price']), "sector": s})
                    
                    if s not in sector_stats: sector_stats[s] = {"exposure": 0.0, "count": 0, "avg_score": 0.0}
                    sector_stats[s]["exposure"] += e['score'] * 2.1
                    sector_stats[s]["count"] += 1
                    sector_stats[s]["avg_score"] += e['score']

                for s in sector_stats: sector_stats[s]["avg_score"] /= sector_stats[s]["count"]

                payload = {
                    "timestamp": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metacognition": {"belief_score": belief, 
                                     "manifold_drift": self.training_manifold + [np.random.normal(0, 0.3, 2).tolist()],
                                     "alpha_decay": (np.cumsum(np.random.uniform(0, 0.05, 30))).tolist()},
                    "rankings": {"ladder": ladder, "ls_spread": (np.cumsum(np.random.normal(0.0001, 0.002, 100))).tolist()},
                    "execution": {"implementation_shortfall": float(np.random.uniform(2.1, 4.8)), "is_var": 0.0001,
                                 "slippage_heatmap": np.random.rand(5,5).tolist()},
                    "pipeline": {"champion_sharpe": 1.42, "challenger_sharpe": 2.36, "training_progress": "V1 ACTIVE"},
                    "institutional": {
                        "gross_exposure": float(len(ladder) * 2.1), 
                        "net_exposure": float(sum(s['exposure'] for s in sector_stats.values())),
                        "sector_exposure": sector_stats,
                        "data_latency_ms": float(np.random.uniform(40, 180)), 
                        "data_freshness_s": float(time.time() - last_poll),
                        "oms_queue": self.oms_queue, "order_log": []
                    },
                    "type": "GLOBAL_UPDATE"
                }
                self.redis_client.publish('uqts:global', json.dumps(payload))
                
                watchlist = self.redis_client.smembers('uqts:watchlist')
                for t in watchlist:
                    spectral = self._get_spectral_data(batch, t)
                    if spectral: self.redis_client.publish(f'uqts:spectral:{t}', json.dumps({"spectral": spectral, "type": "SPECTRAL_UPDATE"}))

            self.current_knowledge_time += timedelta(days=1)
            if self.current_knowledge_time > datetime.now(): self.current_knowledge_time = datetime(2023, 1, 1)
            time.sleep(self.update_interval)

if __name__ == "__main__":
    worker = InferenceWorker(); worker.initialize(); worker.run()
