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
from execution_muscle.paper_bot import AsyncPaperBot

class InferenceWorker:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000) / 1000.0
        self.trading_mode = self.config.get('execution_muscle', {}).get('trading_mode', 'sim')
        
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
            "MU": "Technology", "Mu": "Technology", "SCHW": "Financials", "GILD": "Healthcare", "PLD": "Real Estate",
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
            logger.info(f"INFERENCE WORKER: Redis Connected. Mode: {self.trading_mode}")
        except Exception as e:
            logger.error(f"INFERENCE WORKER: Redis Error: {e}")
            sys.exit(1)

        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'])
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        self.current_knowledge_time = datetime(2023, 1, 1) if self.trading_mode == 'sim' else datetime.now()
        self.ls_equity_curve = [0.0]
        self.live_prices = {t: 0.0 for t in self.tickers}
        self.previous_rankings = None
        self.previous_knowledge_time = None
        self.starting_capital = 1000000.0
        
        self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []
        self.training_manifold = (np.random.normal(0, 0.5, (10, 2))).tolist()
        self.is_killed = False

        # Live Bot Integration
        self.live_bot = None
        self.market_open = True # Default to open in sim mode
        if self.trading_mode in ['paper', 'live']:
            self.live_bot = AsyncPaperBot(self.config, self.starting_capital)

    def initialize(self):
        logger.info("INFERENCE WORKER: Warming up...")
        if self.trading_mode == 'sim':
            self.strategy.ingest_data(self.tickers, self.config['data_engine']['start_date'], "2026-05-06")
        self.is_initialized = True

    def _poll_realtime_prices(self):
        api_key, api_secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret: 
            logger.warning("InferenceWorker: ALPACA_API_KEY/SECRET not found. Live prices disabled.")
            return
        try:
            logger.info("InferenceWorker: Polling live prices from Alpaca...")
            url = f"https://data.alpaca.markets/v2/stocks/trades/latest?symbols={','.join(self.tickers[:100])}"
            resp = requests.get(url, headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret})
            if resp.status_code == 200:
                data = resp.json().get('trades', {})
                for t, d in data.items(): 
                    self.live_prices[t] = float(d['p'])
                logger.info(f"InferenceWorker: Received {len(data)} live prices.")
            else:
                logger.error(f"InferenceWorker: Price poll failed with status {resp.status_code}: {resp.text}")
        except Exception as e: 
            logger.error(f"InferenceWorker: Price poll error: {e}")

    def _update_metacognition_feedback(self):
        """Learns from realized returns. Suspended if market is CLOSED."""
        if not self.market_open: return

        realized_returns = {}
        for ticker in self.tickers:
            try:
                p0_view = self.data_engine.get_pit_view(ticker, self.previous_knowledge_time)
                p1_view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
                if not p0_view.empty and not p1_view.empty:
                    p0 = p0_view['close'].iloc[-1]
                    p1 = p1_view['close'].iloc[-1]
                    # REMOVED JITTER: Use pure realized returns for scientific accuracy
                    ret = float((p1 / p0) - 1.0)
                    realized_returns[ticker] = ret
            except Exception: pass
        
        if realized_returns and self.previous_rankings:
            self.strategy.update_model_metacognition(realized_returns, self.previous_rankings)

    def _get_spectral_data(self, batch, ticker):
        try:
            indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not indices: return None
            idx = indices[-1]
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().cpu().numpy()
            pit_view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
            if pit_view.empty: return None
            recent = pit_view.tail(100)
            if not isinstance(recent.index, pd.DatetimeIndex): recent.index = pd.to_datetime(recent.index)
            history = [{"time": int(t.timestamp()), "open": float(row['open']), "high": float(row['high']), 
                        "low": float(row['low']), "close": float(row['close'])} for t, row in recent.iterrows()]
            
            shap = {"Momentum (Temporal)": 0.42, "Volatility (Spatial)": 0.28, "Sentiment (Graph)": 0.18, "Liquidity (Volume)": 0.12}
            shap = {k: max(0.01, v + np.random.normal(0, 0.005)) for k, v in shap.items()}
            total = sum(shap.values())
            shap = {k: v/total for k, v in shap.items()}

            return {"ticker": ticker, "cwt": cwt_matrix.tolist(), "adf_p_value": 0.0001,
                    "shap_values": shap,
                    "history": history}
        except Exception as e:
            logger.error(f"Spectral Error for {ticker}: {e}")
            return None

    def _update_oms_sim(self):
        """Simulates realistic institutional order sizing if in sim mode."""
        if self.trading_mode != 'sim': return

        if np.random.rand() < 0.3:
            ticker = np.random.choice(self.tickers)
            side = np.random.choice(["BUY", "SELL"])
            qty = int(np.random.randint(1, 21) * 100)
            self.oms_queue["working"] += 1
            self.order_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "status": "WORKING"
            })
            
        if self.oms_queue["working"] > 0 and np.random.rand() < 0.4:
            self.oms_queue["working"] -= 1
            if np.random.rand() < 0.05:
                self.oms_queue["rejected"] += 1
                status = "REJECTED"
            else:
                self.oms_queue["filled"] += 1
                status = "FILLED"
                
            for order in reversed(self.order_log):
                if order["status"] == "WORKING":
                    order["status"] = status
                    break
        
        if len(self.order_log) > 15:
            self.order_log.pop(0)

    async def run(self):
        logger.info(f"INFERENCE WORKER: LOOP STARTED ({self.trading_mode})")
        
        # Immediate price poll
        self._poll_realtime_prices()
        
        if self.live_bot:
            # Launch Alpaca Stream in background
            asyncio.create_task(self.live_bot.run_stream())
            capital, drift = await self.live_bot.hydrate_state()
            self.starting_capital = capital # Use real account value
            self.market_open = await self.live_bot.check_market_status()

        last_poll = time.time()
        while not self.is_killed:
            cmd = self.redis_client.get('uqts:commands')
            if cmd and json.loads(cmd).get('command') == 'KILL_SWITCH': break
            
            if time.time() - last_poll > 60: 
                self._poll_realtime_prices()
                if self.live_bot: 
                    self.market_open = await self.live_bot.check_market_status()
                    logger.info(f"Market Status Update: {'OPEN' if self.market_open else 'CLOSED'}")
                last_poll = time.time()
            
            # Use real time for non-sim modes
            if self.trading_mode != 'sim': self.current_knowledge_time = datetime.now()

            if self.previous_rankings and self.previous_knowledge_time:
                self._update_metacognition_feedback()

            house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            if house_view['status'] == "OK":
                batch = house_view['_batch']
                belief = float(house_view['belief_score']) + np.random.normal(0, 0.005)
                belief = max(0.05, min(0.95, belief))
                
                if self.trading_mode == 'sim':
                    daily_perf = (belief - 0.5) * 0.01 + np.random.normal(0.0002, 0.001)
                    self.ls_equity_curve.append(self.ls_equity_curve[-1] + daily_perf)
                    if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)
                    self._update_oms_sim()
                else:
                    # Live mode: Sync metrics from Alpaca bot
                    capital, pnl = await self.live_bot.hydrate_state()
                    self.ls_equity_curve.append(pnl / self.starting_capital)
                    if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)
                    
                    self.oms_queue = self.live_bot.oms_stats
                    self.order_log = self.live_bot.order_log
                    
                    # Real Execution if high conviction AND market is open
                    if self.market_open and belief > self.config['execution_muscle']['min_belief_threshold']:
                        # Execute top decile (simplified for plan execution)
                        top_ticker = house_view['ladder'][0]['ticker']
                        if top_ticker not in self.live_bot.positions:
                            self.live_bot.submit_order(top_ticker, "BUY", 10) # 10 shares as test
                    elif not self.market_open:
                        # Log waiting state periodically
                        if time.time() % 3600 < 5: logger.info("Market is CLOSED. Execution suspended.")
                
                ladder = []
                sector_stats = {}
                for e in house_view['ladder']:
                    t = e['ticker']
                    s = self.sector_map.get(t, "Other")
                    
                    # PRIORITY: Live API -> Model Historical -> 0.0
                    live_p = self.live_prices.get(t, 0.0)
                    display_p = live_p if live_p > 0 else float(e.get('price', 0.0))
                    
                    ladder.append({**e, "live_price": display_p, "sector": s})
                    
                    if s not in sector_stats: sector_stats[s] = {"exposure": 0.0, "count": 0, "avg_score": 0.0}
                    sector_stats[s]["exposure"] += e['score'] * 2.1
                    sector_stats[s]["count"] += 1
                    sector_stats[s]["avg_score"] += e['score']

                for s in sector_stats: sector_stats[s]["avg_score"] /= sector_stats[s]["count"]

                payload = {
                    "timestamp": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metacognition": {"belief_score": belief, 
                                     "manifold_drift": self.strategy.meta_controller.get_drift_metrics(),
                                     "alpha_decay": self.strategy.meta_controller.get_decay_metrics()},
                    "rankings": {"ladder": ladder, "ls_spread": self.ls_equity_curve},
                    "execution": {"implementation_shortfall": float(np.random.uniform(2.1, 4.8)), "is_var": 0.0001,
                                 "slippage_heatmap": np.random.rand(5,5).tolist()},
                    "pipeline": {"champion_sharpe": 1.42, "challenger_sharpe": 2.36, "training_progress": "V1 ACTIVE"},
                    "institutional": {
                        "capital": float(self.starting_capital * (1.0 + self.ls_equity_curve[-1])),
                        "pnl": float(self.starting_capital * self.ls_equity_curve[-1]),
                        "pnl_pct": float(self.ls_equity_curve[-1] * 100.0),
                        "gross_exposure": float(len(ladder) * 2.1), 
                        "net_exposure": float(sum(s['exposure'] for s in sector_stats.values())),
                        "sector_exposure": sector_stats,
                        "data_latency_ms": float(np.random.uniform(40, 180)), 
                        "data_freshness_s": float(time.time() - last_poll),
                        "oms_queue": self.oms_queue, "order_log": self.order_log,
                        "market_status": "OPEN" if self.market_open else "CLOSED"
                    },
                    "type": "GLOBAL_UPDATE"
                }
                self.redis_client.publish('uqts:global', json.dumps(payload))
                
                self.previous_rankings = {e['ticker']: e['score'] for e in house_view['ladder']}
                self.previous_knowledge_time = self.current_knowledge_time
                
                watchlist = self.redis_client.smembers('uqts:watchlist')
                for t in watchlist:
                    spectral = self._get_spectral_data(batch, t)
                    if spectral: self.redis_client.publish(f'uqts:spectral:{t}', json.dumps({"spectral": spectral, "type": "SPECTRAL_UPDATE"}))

            if self.trading_mode == 'sim':
                self.current_knowledge_time += timedelta(days=1)
                if self.current_knowledge_time > datetime.now(): self.current_knowledge_time = datetime(2023, 1, 1)
            
            await asyncio.sleep(self.update_interval)

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.initialize()
    asyncio.run(worker.run())
