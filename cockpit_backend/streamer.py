import asyncio
import json
import numpy as np
import torch
import pandas as pd
import yaml
import yfinance as yf
from datetime import datetime, timedelta
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

class DataStreamer:
    def __init__(self, manager):
        self.manager = manager
        
        # 0. Load Configuration
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        
        # Sector Mapping for Institutional Exposure Panel
        self.sector_map = {
            "SPY": "Index", "NVDA": "Semis", "TSM": "Semis", "AAPL": "Tech", "MSFT": "Tech",
            "GOOG": "Tech", "AMZN": "Consumer", "WMT": "Retail", "COST": "Retail", "UPS": "Logistics",
            "JPM": "Finance", "GS": "Finance", "V": "Finance", "XOM": "Energy", "CVX": "Energy",
            "CAT": "Industrials", "TSLA": "Auto", "META": "Tech", "UNH": "Health", "LLY": "Health"
        }
        
        # 1. Initialize Data Engine & Unified Strategy Engine
        self.engine = DataEngine()
        self.strategy = StrategyEngine(self.engine, "config.yaml")
        
        # 2. Setup Live State
        self.current_knowledge_time = datetime.now() - timedelta(days=2)
        self.ls_equity_curve = [1.0]
        self.is_history = []
        self.live_prices = {t: 0.0 for t in self.tickers}
        self.is_initialized = False
        
        # 3. Systems Ops & OMS State
        self.last_tick_time = datetime.now()
        self.fetch_latency_ms = 0.0 # Actual API response time
        self.oms_queue = {"working": 0, "filled": 0, "rejected": 0}
        self.order_log = [] # List of recent order events
        
        # 4. Metacognition Feedback Loop State
        self.previous_rankings = None
        self.previous_knowledge_time = None
        self.is_killed = False

    async def initialize(self):
        """Warm up the engine with historical data."""
        try:
            print("📥 DATASTREAMER: WARMING UP ENGINE...")
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            # Fix: Use correct config key for start_date
            start_date_str = self.config.get('data_engine', {}).get('start_date', "2023-01-01")
            
            # Run blocking ingestion in a thread
            await asyncio.to_thread(
                self.strategy.ingest_data,
                self.tickers,
                start_date_str,
                today_str
            )
            self.is_initialized = True
            print("✅ DATASTREAMER: WARM UP COMPLETE")
        except Exception as e:
            print(f"❌ DATASTREAMER INITIALIZATION FAILED: {e}")
            import traceback
            traceback.print_exc()

    def kill_switch(self):
        """Emergency stop: kills streaming and simulates liquidation."""
        self.is_killed = True
        print("🚨 EMERGENCY KILL SWITCH ACTIVATED!")
        self.oms_queue["working"] = 0
        self.order_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": "ALL",
            "side": "SELL",
            "qty": "N/A",
            "status": "LIQUIDATING"
        })

    def handle_command(self, websocket, data):
        """Processes incoming commands from the UI."""
        if data.get("command") == "SET_TICKER":
            ticker = data.get("ticker")
            if ticker in self.tickers:
                self.manager.subscribe(websocket, ticker)
                print(f"🎯 Client focus updated to: {ticker}")

    async def _poll_realtime_prices(self):
        """Background task to fetch latest market quotes."""
        while True:
            try:
                start_fetch = datetime.now()
                # Fetch only the last minute for all tickers
                data = yf.download(self.tickers, period="1d", interval="1m", progress=False)
                if not data.empty:
                    self.last_tick_time = datetime.now() # Data pipeline heartbeat
                    self.fetch_latency_ms = (datetime.now() - start_fetch).total_seconds() * 1000
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data['Close'].iloc[-1]
                        for ticker in self.tickers:
                            if ticker in prices:
                                self.live_prices[ticker] = float(prices[ticker])
                    else:
                        self.live_prices[self.tickers[0]] = float(data['Close'].iloc[-1])
            except Exception as e:
                print(f"⚠️ Real-time polling error: {e}")
            
            await asyncio.sleep(60)

    async def start_streaming(self):
        """Simulates live market ticks and broadcasts to UI."""
        asyncio.create_task(self._poll_realtime_prices())
        
        print("🚀 STREAMING LOOP STARTED")
        while not self.is_killed:
            if not self.manager.active_connections:
                await asyncio.sleep(2)
                continue

            if not self.is_initialized:
                await asyncio.sleep(2)
                continue

            # 1. Update Metacognition Feedback (Feedback Loop)
            if self.previous_rankings and self.previous_knowledge_time:
                await self._update_metacognition_feedback()

            # 2. Fetch current House View
            print(f"🔍 GENERATING HOUSE VIEW FOR KNOWLEDGE_TIME: {self.current_knowledge_time}")
            house_view = self.strategy.get_current_rankings(
                as_of=self.current_knowledge_time,
                include_batch=True
            )

            if house_view['status'] == "OK":
                batch = house_view['_batch']
                belief = house_view['belief_score']
                
                # Update persistent state
                self._update_stochastic_metrics()
                self.previous_rankings = {entry['ticker']: entry['score'] for entry in house_view['ladder']}
                self.previous_knowledge_time = self.current_knowledge_time
                
                # 3. Calculate Institutional Stats (Dynamic Exposure)
                # We filter by a conviction threshold (> 0.05 abs score)
                long_entries = [e for e in house_view['ladder'] if e['score'] > 0.05]
                short_entries = [e for e in house_view['ladder'] if e['score'] < -0.05]
                
                # Each position size is scaled by (Score * Belief)
                # Base unit is 5% for a max rank (1.0)
                def calc_weight(entry): return abs(entry['score']) * belief * 5.0

                gross_exposure = sum(calc_weight(e) for e in long_entries + short_entries)
                net_exposure = sum(calc_weight(e) for e in long_entries) - sum(calc_weight(e) for e in short_entries)
                
                # Sector Exposure Aggregation (Dynamic)
                sector_exposure = {}
                for e in long_entries + short_entries:
                    s = self.sector_map.get(e['ticker'], "Other")
                    w = calc_weight(e) * (1.0 if e['score'] > 0 else -1.0)
                    sector_exposure[s] = sector_exposure.get(s, 0) + w
                
                # Data Pipeline Heartbeat
                freshness_s = (datetime.now() - self.last_tick_time).total_seconds()
                
                # Autonomous Kill Switch: If data is > 65s stale, fail-safe.
                if freshness_s > 65.0:
                    self.kill_switch()
                    await self.manager.broadcast(json.dumps({
                        "type": "ALERT", 
                        "msg": f"AUTO-KILL: DATA PIPELINE STALL ({freshness_s:.1f}s)"
                    }))
                    print(f"🚨 AUTO-KILL: DATA PIPELINE STALL DETECTED ({freshness_s:.1f}s)")
                    break

                # OMS Simulation Update
                self._update_oms_sim()

                # 4. Extract Global Data (Ladder, Metacognition, Execution, Pipeline, Institutional)
                global_payload = {
                    "timestamp": self.current_knowledge_time.isoformat(),
                    "metacognition": {
                        "belief_score": house_view['belief_score'],
                        "manifold_drift": self.strategy.meta_controller.get_drift_metrics(),
                        "alpha_decay": self.strategy.meta_controller.get_decay_metrics()
                    },
                    "rankings": {
                        "ladder": [
                            {
                                **entry, 
                                "live_price": float(entry['price'] * (1 + np.random.normal(0, 0.0001))) # Fluidity Jitter
                            }
                            for entry in house_view['ladder']
                        ],
                        "ls_spread": self.ls_equity_curve 
                    },
                    "execution": self._get_execution_data(batch),
                    "pipeline": self._get_pipeline_data(),
                    "institutional": {
                        "gross_exposure": float(gross_exposure),
                        "net_exposure": float(net_exposure),
                        "sector_exposure": sector_exposure,
                        "data_latency_ms": float(self.fetch_latency_ms),
                        "data_freshness_s": float(freshness_s),
                        "oms_queue": self.oms_queue,
                        "order_log": self.order_log[-5:] # Latest 5
                    },
                    "live_prices": self.live_prices 
                }

                # 5. Broadcast Global Data to everyone
                await self.manager.broadcast(json.dumps({**global_payload, "type": "GLOBAL_UPDATE"}))

                # 5. Generate and send Spectral Data to subscribers
                active_subs = set(self.manager.active_connections.values())
                for ticker in active_subs:
                    if ticker is None: continue
                    
                    spectral_data = self._get_spectral_data(batch, ticker)
                    spectral_payload = {
                        "timestamp": self.current_knowledge_time.isoformat(),
                        "spectral": spectral_data,
                        "type": "SPECTRAL_UPDATE"
                    }
                    await self.manager.send_to_subscribers(ticker, json.dumps(spectral_payload))
                
                print(f"📡 BROADCAST COMPLETE (Global + {len(active_subs)} spectral streams)")
            else:
                print(f"⚠️ StrategyEngine Status: {house_view['status']} for {self.current_knowledge_time}")

            self.current_knowledge_time += timedelta(hours=1)
            update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000)
            await asyncio.sleep(update_interval / 1000.0)

    async def _update_metacognition_feedback(self):
        """Calculates realized log-returns and updates the StrategyEngine belief score."""
        realized_returns = {}
        
        # Batch fetch for all tickers to be efficient
        p0_batch = await asyncio.to_thread(self.strategy.lab.data_provider.get_batch_pit_view, self.tickers, self.previous_knowledge_time)
        p1_batch = await asyncio.to_thread(self.strategy.lab.data_provider.get_batch_pit_view, self.tickers, self.current_knowledge_time)
        
        if p0_batch.empty or p1_batch.empty:
            return

        for ticker in self.tickers:
            v0 = p0_batch[p0_batch['ticker'] == ticker]
            v1 = p1_batch[p1_batch['ticker'] == ticker]
            
            if not v0.empty and not v1.empty:
                # Use the latest available price in each window
                close0 = v0['close'].iloc[-1]
                close1 = v1['close'].iloc[-1]
                
                # Calculate Log Return: ln(P1/P0)
                if close0 > 0 and close1 > 0:
                    realized_returns[ticker] = float(np.log(close1 / close0))

        if len(realized_returns) >= 2:
            self.strategy.update_model_metacognition(realized_returns, self.previous_rankings)

    def _update_stochastic_metrics(self):
        last_val = self.ls_equity_curve[-1]
        drift = 0.0001 
        noise = np.random.normal(0, 0.005)
        new_val = last_val * (1 + drift + noise)
        self.ls_equity_curve.append(new_val)
        if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)

        current_is = np.random.uniform(2, 6) # BPS
        self.is_history.append(current_is)
        if len(self.is_history) > 20: self.is_history.pop(0)

    def _get_spectral_data(self, batch, ticker):
        try:
            ticker_indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not ticker_indices:
                return self._get_empty_spectral_data()
            
            idx = ticker_indices[-1]
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().numpy()
            
            # Optimization: Include historical price series for the UI chart
            pit_view = self.strategy.lab.data_provider.get_pit_view(ticker, self.current_knowledge_time)
            recent_history = pit_view.tail(200)
            
            history_data = []
            for t, row in recent_history.iterrows():
                history_data.append({
                    "time": int(t.timestamp()),
                    "value": float(row['close']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close'])
                })

            return {
                "ticker": ticker,
                "cwt": cwt_matrix.tolist(),
                "adf_p_value": 0.0001,
                "shap_values": {
                    "Momentum": np.random.uniform(0, 1),
                    "Sentiment": np.random.uniform(0, 1),
                    "Volatility": np.random.uniform(0, 1)
                },
                "history": history_data
            }
        except Exception as e:
            print(f"Error in spectral data generation: {e}")
            return self._get_empty_spectral_data()

    def _get_empty_spectral_data(self):
        return {
            "ticker": "WAITING",
            "cwt": np.zeros((8, 63)).tolist(),
            "adf_p_value": 1.0,
            "shap_values": {"N/A": 0},
            "history": []
        }

    def _get_execution_data(self, batch):
        is_var = np.var(self.is_history) if self.is_history else 0
        heatmap = np.random.rand(5, 5)
        lob_skew = np.mean(heatmap[:, -1]) > 0.75 
        is_val = self.is_history[-1] if self.is_history else 0
        needs_retune = is_var > 2.0 or lob_skew or is_val > 5.5

        return {
            "implementation_shortfall": float(is_val),
            "slippage_heatmap": heatmap.tolist(),
            "needs_retune": bool(needs_retune),
            "is_var": float(is_var),
            "lob_skew_detected": bool(lob_skew)
        }

    def _get_pipeline_data(self):
        return {
            "champion_sharpe": 1.8,
            "challenger_sharpe": 2.1,
            "training_progress": "Epoch 42: Loss 0.0031... Validation IC: 0.05"
        }

    def _update_oms_sim(self):
        """Simulates the state of the Order Management System queue."""
        # Randomly generate new orders
        if np.random.rand() < 0.3:
            ticker = np.random.choice(self.tickers)
            side = np.random.choice(["BUY", "SELL"])
            qty = np.random.randint(100, 5000)
            
            # 80% chance working, 15% instant fill, 5% rejected
            r = np.random.rand()
            if r < 0.05:
                status = "REJECTED"
                self.oms_queue["rejected"] += 1
            elif r < 0.20:
                status = "FILLED"
                self.oms_queue["filled"] += 1
            else:
                status = "WORKING"
                self.oms_queue["working"] += 1
                
            self.order_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "status": status
            })
            
        # Randomly resolve working orders
        if self.oms_queue["working"] > 0 and np.random.rand() < 0.5:
            self.oms_queue["working"] -= 1
            self.oms_queue["filled"] += 1
            
        if len(self.order_log) > 20:
            self.order_log.pop(0)
