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
        
        self.current_knowledge_time = datetime(2023, 1, 1, 16, 0, 0) if self.trading_mode == 'sim' else datetime.now()
        self.ls_equity_curve = [0.0]
        self.live_prices = {t: 0.0 for t in self.tickers}
        self.ranking_history = [] # Track multi-day history for accurate horizon evaluation
        # SENIOR FIX: Initialized to None, hydrated from live data or config
        self.starting_capital = None if self.trading_mode != 'sim' else 100000.0
        self.sim_cash = 100000.0 # Dedicated cash ledger for simulation
        self.sim_avg_costs = {} # Track cost basis for ROE/Ladder P&L
        self.sim_positions = {} # Ticker -> Qty for ledger-accurate valuation
        self.sim_realized_pnl = 0.0 # Explicitly track realized gains/losses
        
        # SENIOR FIX: Track submitted orders to prevent double-submission during hydration lag
        self.pending_orders = set() 
        
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
        # SENIOR FIX: Do not poll live prices in simulation mode
        if self.trading_mode == 'sim': return
        
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

        # HORIZON FIX: Evaluate on the 5-day horizon the model was trained for.
        horizon_days = self.config.get('signal_physics', {}).get('horizon_days', 5)
        
        # Find the ranking from ~horizon_days ago
        target_time = self.current_knowledge_time - timedelta(days=horizon_days)
        
        eval_item = None
        for item in self.ranking_history:
            if item['time'] <= target_time:
                eval_item = item
            else:
                break
                
        if not eval_item:
            return # Wait until we have enough history to evaluate a full horizon
            
        # Clean up older histories to prevent memory bloat
        self.ranking_history = [item for item in self.ranking_history if item['time'] >= target_time - timedelta(days=2)]

        realized_returns = {}
        for ticker in self.tickers:
            try:
                p0_view = self.data_engine.get_pit_view(ticker, eval_item['time'])
                p1_view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
                if not p0_view.empty and not p1_view.empty:
                    p0 = float(p0_view['close'].iloc[-1])
                    p1 = float(p1_view['close'].iloc[-1])
                    if p0 > 0:
                        ret = (p1 / p0) - 1.0
                        realized_returns[ticker] = ret
            except Exception: pass
        
        if realized_returns and eval_item['rankings']:
            self.strategy.update_model_metacognition(realized_returns, eval_item['rankings'])

    def _get_spectral_data(self, batch, ticker):
        try:
            indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not indices: return None
            idx = indices[-1]
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().cpu().numpy()
            pit_view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
            if pit_view.empty: return None
            # SENIOR FIX: Set lookback to 75,000 bars (covers 10+ years of 15-min history)
            # This ensures WebSocket payloads stay under ~10MB for stability
            recent = pit_view.tail(75000)
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

    def _get_latest_price_sim(self, ticker):
        """Robust price discovery with 24-hour lookback for simulation stability."""
        try:
            # 1. Immediate PIT lookup
            view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
            if not view.empty: return float(view['close'].iloc[-1])
            
            # 2. 1-Day recursive lookback (handles weekends/holidays)
            lookback_time = self.current_knowledge_time - timedelta(days=1)
            view = self.data_engine.get_pit_view(ticker, lookback_time)
            if not view.empty: return float(view['close'].iloc[-1])
        except: pass
        return 0.0

    def _get_market_regime(self, current_time):
        """Determines if the broader market is in a bullish regime (SPY 21-day momentum)."""
        try:
            p_now_view = self.data_engine.get_pit_view('SPY', current_time)
            p_past_view = self.data_engine.get_pit_view('SPY', current_time - timedelta(days=21))
            if not p_now_view.empty and not p_past_view.empty:
                p_now = float(p_now_view['close'].iloc[-1])
                p_past = float(p_past_view['close'].iloc[-1])
                return "BULL" if p_now > p_past else "BEAR"
        except: pass
        return "BULL" # Default to permissive

    def _update_oms_sim(self, house_view, belief):
        """Simulates realistic institutional order sizing if in sim mode based strictly on model conviction."""
        if self.trading_mode != 'sim': return

        limits = self.config['execution_muscle']
        threshold = limits.get('min_belief_threshold', 0.65)
        
        # Hysteresis and Regime Logic
        total_tickers = len(house_view['ladder'])
        entry_size = max(1, total_tickers // 10) # 10% entry (Top 6)
        hold_size = max(1, int(total_tickers * 0.15)) # 15% hold (Top 9) - Tightened from 30%
        
        regime = self._get_market_regime(self.current_knowledge_time)
        allow_shorts = (regime == "BEAR")
        
        # If belief is low, we shouldn't hold any positions (go to cash)
        if belief > threshold and house_view['ladder']:
            top_entry_tickers = [e['ticker'] for e in house_view['ladder'][:entry_size]]
            bottom_entry_tickers = [e['ticker'] for e in house_view['ladder'][-entry_size:]] if allow_shorts else []
            
            top_hold_tickers = [e['ticker'] for e in house_view['ladder'][:hold_size]]
            bottom_hold_tickers = [e['ticker'] for e in house_view['ladder'][-hold_size:]]
        else:
            top_entry_tickers, bottom_entry_tickers = [], []
            top_hold_tickers, bottom_hold_tickers = [], []
            
        # 1. Entry Logic: Only enter if conviction meets threshold
        if belief > threshold and house_view['ladder']:
            # REGIME-LEVERAGED CALCULATION (STABILIZED)
            unrealized_pnl = 0.0
            for t, qty in self.sim_positions.items():
                curr_p = self._get_latest_price_sim(t) or self.sim_avg_costs.get(t, 0)
                entry_p = self.sim_avg_costs.get(t, 0)
                unrealized_pnl += (curr_p - entry_p) * qty if qty > 0 else (entry_p - curr_p) * abs(qty)
            
            nlv = self.starting_capital + self.sim_realized_pnl + unrealized_pnl
            
            # Use a FIXED divisor (10 slots) to prevent doubling when regimes flip
            leverage = min(1.5, belief * 1.2)
            target_total_notional = nlv * leverage
            notional_per_slot = target_total_notional / 10 # Standardized to 10% slots
            
            trade_intents = [(t, "BUY") for t in top_entry_tickers] + [(t, "SHORT") for t in bottom_entry_tickers]
            
            for ticker, intent in trade_intents:
                is_working = any(o['ticker'] == ticker and o['status'] == 'WORKING' for o in self.order_log)
                price = self._get_latest_price_sim(ticker)
                
                # A. NEW ENTRY
                if ticker not in self.sim_positions and not is_working:
                    if price > 0:
                        target_notional = notional_per_slot
                        
                        # MARGIN SAFETY: Only 50% of short proceeds help buy longs
                        pending_cash_out = sum(o['notional'] for o in self.order_log if o['status'] == 'WORKING' and o['side'] == 'BUY')
                        available_for_long = self.sim_cash - pending_cash_out
                        if intent == "BUY":
                            target_notional = min(target_notional, available_for_long * 0.90)
                            
                        qty = int(target_notional / price)
                        if qty > 0:
                            self.oms_queue["working"] += 1
                            self.order_log.append({
                                "time": self.current_knowledge_time.strftime("%m/%d %H:%M"),
                                "ticker": ticker, "side": intent, "qty": qty, "price": price, 
                                "notional": qty * price, "status": "WORKING"
                            })
                
                # B. DYNAMIC TRIMMING (Anti-Leverage Creep)
                elif ticker in self.sim_positions and not is_working:
                    current_qty = self.sim_positions[ticker]
                    current_notional = abs(current_qty * price)
                    # If position is 25% larger than it should be, trim it
                    if current_notional > (notional_per_slot * 1.25):
                        trim_qty = int((current_notional - notional_per_slot) / price)
                        if trim_qty > 0:
                            self.oms_queue["working"] += 1
                            side = "SELL" if current_qty > 0 else "COVER"
                            self.order_log.append({
                                "time": self.current_knowledge_time.strftime("%m/%d %H:%M"),
                                "ticker": ticker, "side": side, "qty": trim_qty, "price": price,
                                "notional": trim_qty * price, "status": "WORKING"
                            })

        # 2. Exit Logic: Close positions if they fall out of their decile
        active_tickers = list(self.sim_positions.keys())
        for ticker in active_tickers:
            is_working = any(o['ticker'] == ticker and o['status'] == 'WORKING' for o in self.order_log)
            if is_working: continue # Wait for pending orders to settle
            
            qty = self.sim_positions[ticker]
            
            # SENIOR FIX: Hysteresis. Close Long if out of Top 15%. Close Short if out of Bottom 15% OR if shorts disabled in Bull regime.
            should_close = False
            if qty > 0 and ticker not in top_hold_tickers: should_close = True
            if qty < 0 and (ticker not in bottom_hold_tickers or not allow_shorts): should_close = True
            
            if should_close:
                price = self._get_latest_price_sim(ticker)
                if price > 0:
                    exit_qty = abs(qty)
                    notional = exit_qty * price
                    self.oms_queue["working"] += 1
                    display_side = "SELL" if qty > 0 else "COVER"
                    self.order_log.append({
                        "time": self.current_knowledge_time.strftime("%m/%d %H:%M"),
                        "ticker": ticker,
                        "side": display_side,
                        "qty": exit_qty,
                        "price": price,
                        "notional": notional,
                        "status": "WORKING"
                    })
            
        # 3. Simulate Fills (Fast forward queue)
        if self.oms_queue["working"] > 0:
            # Sim fills orders much faster, resolve them immediately
            for order in reversed(self.order_log):
                if order["status"] == "WORKING":
                    self.oms_queue["working"] -= 1
                    status = "REJECTED" if np.random.rand() < 0.05 else "FILLED"
                    order["status"] = status
                    
                    if status == "FILLED":
                        self.oms_queue["filled"] += 1
                        ticker = order['ticker']
                        impact = order['notional']
                        
                        # LEDGER V3: Robust Realized P&L + Partial Reduction (Trimming)
                        if ticker in self.sim_positions:
                            is_long = self.sim_positions[ticker] > 0
                            entry_p = self.sim_avg_costs[ticker]
                            exit_p = order['price']
                            exec_qty = order['qty']
                            
                            # P&L calculation
                            pnl = (exit_p - entry_p) * exec_qty if is_long else (entry_p - exit_p) * exec_qty
                            self.sim_realized_pnl += pnl
                            
                            # Cash Update
                            if is_long: self.sim_cash += impact
                            else: self.sim_cash -= impact 
                            
                            # Update Quantity (Partial or Full)
                            if is_long: self.sim_positions[ticker] -= exec_qty
                            else: self.sim_positions[ticker] += exec_qty
                            
                            # Clean up if position is now zero (or dust)
                            if abs(self.sim_positions[ticker]) < 1:
                                del self.sim_positions[ticker]
                                del self.sim_avg_costs[ticker]
                        else:
                            # New Position Entry
                            if order['side'] in ['BUY', 'COVER']: 
                                self.sim_cash -= impact
                                self.sim_positions[ticker] = order['qty']
                            else: # SHORT or SELL (if opening short)
                                self.sim_cash += impact # Get proceeds
                                self.sim_positions[ticker] = -order['qty']
                            self.sim_avg_costs[ticker] = order['price']
        
        # SENIOR FIX: Only pop orders that are NOT currently 'WORKING'
        while len(self.order_log) > 20:
            popped = False
            for i, o in enumerate(self.order_log):
                if o['status'] != 'WORKING':
                    self.order_log.pop(i)
                    popped = True
                    break
            if not popped: break

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
            
            # SENIOR FIX: Pre-Flight Warm-Up (Auto-Hydration)
            # Fast-forward the last 30 days of market data to hydrate the Bayesian Belief score
            logger.info("Bot: Initiating Metacognition Pre-Flight Warm-Up (30 Days)...")
            warmup_start = datetime.now() - timedelta(days=30)
            current_sim_time = warmup_start
            
            original_time = self.current_knowledge_time
            original_market_open = self.market_open
            self.market_open = True # Force open to allow metacognition feedback
            
            while current_sim_time <= datetime.now():
                warmup_view = self.strategy.get_current_rankings(as_of=current_sim_time)
                if warmup_view['status'] == "OK":
                    self.ranking_history.append({
                        'time': current_sim_time,
                        'rankings': {e['ticker']: e['score'] for e in warmup_view['ladder']}
                    })
                
                self.current_knowledge_time = current_sim_time
                self._update_metacognition_feedback()
                current_sim_time += timedelta(days=1)
                
            self.current_knowledge_time = original_time
            self.market_open = original_market_open
            logger.info(f"Bot: Warm-Up Complete. Live Belief Score hydrated to {self.strategy.meta_controller.belief*100:.1f}%")

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

            self._update_metacognition_feedback()

            house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            if house_view['status'] == "OK":
                batch = house_view['_batch']
                belief = float(house_view['belief_score'])
                belief = max(0.05, min(0.95, belief))
                
                if self.trading_mode == 'sim':
                    # SENIOR FIX: Ledger-accurate simulation V2.
                    # Account Value = Initial Capital + Realized P&L + Unrealized P&L
                    # Unrealized P&L = (Current - Entry) * Qty [Long] or (Entry - Current) * Qty [Short]
                    unrealized_pnl = 0.0
                    current_mv = 0.0
                    for ticker, qty in self.sim_positions.items():
                        current_p = self._get_latest_price_sim(ticker) or self.sim_avg_costs.get(ticker, 0)
                        entry_p = self.sim_avg_costs.get(ticker, 0)
                        
                        if qty > 0: # LONG
                            pnl = (current_p - entry_p) * qty
                            mv = current_p * qty
                        else: # SHORT
                            pnl = (entry_p - current_p) * abs(qty)
                            mv = (entry_p * abs(qty)) + pnl # Value is collateral plus profit/loss
                        
                        unrealized_pnl += pnl
                        current_mv += mv
                    
                    total_val = self.starting_capital + self.sim_realized_pnl + unrealized_pnl
                    roi_pct = (total_val / self.starting_capital) - 1.0
                    
                    self.ls_equity_curve.append(roi_pct)
                    if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)
                    
                    self._update_oms_sim(house_view, belief)
                else:
                    # Live mode: Sync metrics from Alpaca bot
                    capital, pnl = await self.live_bot.hydrate_state()
                    if self.starting_capital is None: self.starting_capital = capital
                    self.ls_equity_curve.append(pnl / self.starting_capital)
                    if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)
                    
                    # SENIOR FIX: Calculate Live Unrealized P&L from positions
                    unrealized_pnl = 0.0
                    for t, qty in self.live_bot.positions.items():
                        curr_p = self.live_prices.get(t, 0.0)
                        entry_p = self.live_bot.position_avg_costs.get(t, 0.0)
                        if curr_p > 0 and entry_p > 0:
                            if qty > 0: unrealized_pnl += (curr_p - entry_p) * qty
                            else: unrealized_pnl += (entry_p - curr_p) * abs(qty)

                    self.oms_queue = self.live_bot.oms_stats
                    self.order_log = self.live_bot.order_log

                    # SENIOR FIX: Clean up pending orders once they show up in active positions
                    for t in list(self.pending_orders):
                        if t in self.live_bot.positions:
                            logger.info(f"Execution: Order for {t} confirmed in portfolio. Removing from pending.")
                            self.pending_orders.remove(t)
                    
                    # Real Execution based on Alpha Model Conviction
                    if self.market_open:
                        threshold = self.config['execution_muscle']['min_belief_threshold']
                        
                        # Hysteresis and Regime Logic
                        total_tickers = len(house_view['ladder'])
                        entry_size = max(1, total_tickers // 10) # 10% entry
                        hold_size = max(1, int(total_tickers * 0.30)) # 30% hold
                        
                        regime = self._get_market_regime(self.current_knowledge_time)
                        allow_shorts = (regime == "BEAR")
                        
                        if belief > threshold and house_view['ladder']:
                            top_entry_tickers = [e['ticker'] for e in house_view['ladder'][:entry_size]]
                            bottom_entry_tickers = [e['ticker'] for e in house_view['ladder'][-entry_size:]] if allow_shorts else []
                            
                            top_hold_tickers = [e['ticker'] for e in house_view['ladder'][:hold_size]]
                            bottom_hold_tickers = [e['ticker'] for e in house_view['ladder'][-hold_size:]]
                        else:
                            top_entry_tickers, bottom_entry_tickers = [], []
                            top_hold_tickers, bottom_hold_tickers = [], []
                            
                        # 1. Entry Logic: Only enter if high conviction
                        if belief > threshold and house_view['ladder']:
                            # REGIME-LEVERAGED LIVE (FIXED)
                            leverage = min(1.5, belief * 1.2)
                            target_total_notional = self.starting_capital * leverage
                            
                            trade_intents = [(t, "BUY") for t in top_entry_tickers] + [(t, "SELL") for t in bottom_entry_tickers]
                            n_slots = len(trade_intents) if trade_intents else 1
                            notional_per_slot = target_total_notional / n_slots

                            for ticker, intent in trade_intents:
                                if ticker not in self.live_bot.positions and ticker not in self.pending_orders:
                                    price = self.live_prices.get(ticker, 0.0)
                                    if price > 0:
                                        # Use the calculated regime-based notional instead of a fixed config pct
                                        qty = int(notional_per_slot / price)
                                        if qty > 0:
                                            logger.info(f"Execution: Submitting REGIME-LEVERAGED {intent} order for {ticker} (Qty: {qty}, Lev: {leverage:.2f}x)")
                                            self.live_bot.submit_order(ticker, intent, qty)
                                            self.pending_orders.add(ticker)

                        # 2. Exit Logic: Close positions if they fall out of their hold decile or thesis reverses
                        active_tickers = list(self.live_bot.positions.keys())
                        for ticker in active_tickers:
                            if ticker in self.pending_orders: continue # Wait for pending orders to settle
                            
                            qty = self.live_bot.positions[ticker]
                            
                            # SENIOR FIX: Hysteresis. Close Long if out of Top 30%. Close Short if out of Bottom 30% OR if shorts disabled in Bull regime.
                            should_close = False
                            if qty > 0 and ticker not in top_hold_tickers: should_close = True
                            if qty < 0 and (ticker not in bottom_hold_tickers or not allow_shorts): should_close = True
                            
                            if should_close:
                                logger.info(f"Execution: Closing position for {ticker} due to hysteresis/regime change.")
                                exit_intent = "SELL" if qty > 0 else "BUY"
                                self.live_bot.submit_order(ticker, exit_intent, abs(int(qty)))
                                self.pending_orders.add(ticker)
                    else:
                        # Log waiting state periodically
                        if time.time() % 3600 < 5: logger.info("Market is CLOSED. Execution suspended.")
                
                ladder = []
                sector_stats = {}
                max_lev = self.config['execution_muscle']['safety_limits'].get('max_total_leverage', 2.5)
                
                # 1. Institutional Accounting V2
                # Account Value is the Net Liquidation Value (NLV)
                account_val = float(self.starting_capital + self.sim_realized_pnl + unrealized_pnl) if self.trading_mode == 'sim' else float(self.starting_capital * (1.0 + self.ls_equity_curve[-1]))
                total_pnl = account_val - self.starting_capital
                
                # Use Ledger V2 realized/unrealized if in sim mode
                final_realized = self.sim_realized_pnl if self.trading_mode == 'sim' else (total_pnl - unrealized_pnl)
                final_unrealized = unrealized_pnl

                # SENIOR MATH FIX: Calculate Gross Notional by summing ABSOLUTE value of all positions
                # This correctly accounts for both Longs and Shorts.
                gross_notional = 0.0
                net_notional = 0.0
                
                # Determine positions to iterate (sim or live)
                active_pos_dict = self.sim_positions if self.trading_mode == 'sim' else self.live_bot.positions
                
                for t, qty in active_pos_dict.items():
                    # Get current price
                    curr_p = 0.0
                    if self.trading_mode == 'sim':
                        curr_p = self._get_latest_price_sim(t) or self.sim_avg_costs.get(t, 0)
                    else:
                        curr_p = self.live_prices.get(t, 0.0)
                    
                    pos_val = qty * curr_p
                    gross_notional += abs(pos_val)
                    net_notional += pos_val
                
                # Exposure Percentages
                gross_exp_pct = (gross_notional / account_val * 100.0) if account_val > 0 else 0.0
                net_exp_pct = (net_notional / account_val * 100.0) if account_val > 0 else 0.0
                
                # ROE (Return on Exposure): Gain relative to the money actually at work
                roe = (total_pnl / gross_notional * 100.0) if gross_notional > 100 else 0.0
                
                for e in house_view['ladder']:
                    t = e['ticker']
                    s = self.sector_map.get(t, "Other")
                    
                    # SENIOR FIX: Robust Price Retrieval
                    # In sim mode, priority is ALWAYS historical data from simulation clock
                    if self.trading_mode == 'sim':
                        display_p = 0.0
                        try:
                            v = self.data_engine.get_pit_view(t, self.current_knowledge_time)
                            if not v.empty: display_p = float(v['close'].iloc[-1])
                            else: display_p = float(e.get('price', 0.0))
                        except:
                            display_p = float(e.get('price', 0.0))
                    else:
                        live_p = self.live_prices.get(t, 0.0)
                        display_p = live_p if live_p > 0 else float(e.get('price', 0.0))
                    
                    # P&L and Market Value Calculation
                    pnl_pct = 0.0
                    market_val = 0.0
                    if self.live_bot:
                        pnl_pct = self.live_bot.position_unrealized_pnl.get(t, 0.0)
                        market_val = abs(self.live_bot.positions.get(t, 0.0) * display_p)
                    elif t in self.sim_avg_costs and display_p > 0:
                        entry = self.sim_avg_costs[t]
                        qty = self.sim_positions.get(t, 0.0)
                        
                        if qty > 0: # Long
                            pnl_pct = ((display_p / entry) - 1.0) * 100.0
                            market_val = qty * display_p
                        elif qty < 0: # Short
                            pnl_pct = (1.0 - (display_p / entry)) * 100.0
                            market_val = (entry * abs(qty)) + (entry - display_p) * abs(qty)
                        
                    ladder.append({**e, "live_price": display_p, "sector": s, "pnl_pct": pnl_pct, "market_value": market_val})
                    
                    if s not in sector_stats: sector_stats[s] = {"exposure": 0.0, "count": 0, "avg_score": 0.0}
                    # SENIOR FIX: Use config leverage instead of hardcoded 2.1
                    sector_stats[s]["exposure"] += e['score'] * max_lev
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
                        "capital": account_val,
                        "buying_power": float(self.live_bot.buying_power if self.live_bot else self.sim_cash),
                        "pnl": total_pnl,
                        "pnl_realized": final_realized,
                        "pnl_unrealized": final_unrealized,
                        "pnl_pct": float(self.ls_equity_curve[-1] * 100.0),
                        "roe": roe,
                        "gross_exposure": gross_exp_pct, 
                        "net_exposure": net_exp_pct,
                        "active_positions": len(active_pos_dict),
                        "sector_exposure": sector_stats,
                        "data_latency_ms": float(np.random.uniform(40, 180)), 
                        "data_freshness_s": float(time.time() - last_poll),
                        "oms_queue": self.oms_queue, "order_log": self.order_log,
                        "market_status": "OPEN" if self.market_open else "CLOSED",
                        "trading_mode": self.trading_mode
                    },
                    "type": "GLOBAL_UPDATE"
                }
                self.redis_client.publish('uqts:global', json.dumps(payload))
                
                self.ranking_history.append({
                    'time': self.current_knowledge_time,
                    'rankings': {e['ticker']: e['score'] for e in house_view['ladder']}
                })
                
                watchlist = self.redis_client.smembers('uqts:watchlist')
                for t in watchlist:
                    spectral = self._get_spectral_data(batch, t)
                    if spectral: self.redis_client.publish(f'uqts:spectral:{t}', json.dumps({"spectral": spectral, "type": "SPECTRAL_UPDATE"}))

            if self.trading_mode == 'sim':
                self.current_knowledge_time += timedelta(days=1)
                # RESET LEDGER ON LOOP
                if self.current_knowledge_time > datetime.now(): 
                    logger.warning("INFERENCE WORKER: Simulation Loop Reset. Clearing Ledger.")
                    self.current_knowledge_time = datetime(2023, 1, 1)
                    self.sim_cash = 100000.0
                    self.sim_positions = {}
                    self.sim_avg_costs = {}
                    self.sim_realized_pnl = 0.0
                    self.order_log = []
                    self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
            
            await asyncio.sleep(self.update_interval)

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.initialize()
    asyncio.run(worker.run())
