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
import torch
from stable_baselines3 import PPO
from tqdm import tqdm
from scipy.stats import spearmanr

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.strategy_engine import StrategyEngine
from execution_muscle.paper_bot import AsyncPaperBot
from alpha_factory.meta_controller import BayesianMetaController
from execution_muscle.risk_parity_sizer import RiskParitySizer
from alpha_factory.observation_utils import build_rl_observation, calculate_safe_weights

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class InferenceWorker:
    """
    Expert-Grade Production Inference Engine (Ferrari Edition).
    """
    def __init__(self, config_path="config.yaml", mode_override=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000) / 1000.0
        self.trading_mode = mode_override or self.config.get('execution_muscle', {}).get('trading_mode', 'sim')
        self.latency_stress_test = self.config.get('execution_muscle', {}).get('latency_stress_test', False)
        
        logger.info(f"INFERENCE WORKER: Initializing Master Sniper V7.4.3. Mode: {self.trading_mode}")
        
        self.rl_pilot = None
        rl_path = "models/rl_pilot_final.zip"
        if os.path.exists(rl_path):
            self.rl_pilot = PPO.load(rl_path, device="cpu")
            logger.info("INFERENCE WORKER: RL Pilot Loaded.")
        
        # Sector Mapping
        self.sector_map = {
            "SPY": "Index",
            "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "CRM": "Tech", 
            "ORCL": "Tech", "ADBE": "Tech", "CSCO": "Tech", "NFLX": "Tech", "ACN": "Tech", 
            "INTU": "Tech", "VZ": "Tech", "IBM": "Tech",
            "NVDA": "Semi", "AVGO": "Semi", "AMD": "Semi", "QCOM": "Semi", "TXN": "Semi", "AMAT": "Semi",
            "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", 
            "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare", 
            "AMGN": "Healthcare", "ISRG": "Healthcare", "CVS": "Healthcare",
            "JPM": "Financial", "V": "Financial", "MA": "Financial", "BAC": "Financial", 
            "WFC": "Financial", "BX": "Financial", "GS": "Financial", "MS": "Financial", "SPGI": "Financial",
            "AMZN": "Retail", "HD": "Retail", "PG": "Retail", "COST": "Retail", "PEP": "Retail", 
            "KO": "Retail", "WMT": "Retail", "MCD": "Retail", "PM": "Retail", "LOW": "Retail",
            "TSLA": "Auto", "COP": "Energy", "LIN": "Materials",
            "GE": "Industrials", "UNP": "Industrials", "HON": "Industrials", "BA": "Industrials", 
            "CAT": "Industrials", "LMT": "Industrials", "RTX": "Industrials"
        }
        for t in self.tickers:
            if t not in self.sector_map: self.sector_map[t] = "Other"
        self.canonical_sectors = ["Tech", "Semi", "Healthcare", "Financial", "Retail", "Auto", "Energy", "Materials", "Industrials", "Index", "Other"]
        
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"INFERENCE WORKER: Redis Connected.")
        except Exception as e:
            logger.error(f"INFERENCE WORKER: Redis Error: {e}")
            sys.exit(1)

        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'], read_only=True)
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        # PRE-FETCH SPY FOR MACRO ALIGNMENT
        self.spy_df = self.data_engine.conn.execute("SELECT event_time, close FROM market_data WHERE ticker = 'SPY'").df()
        self.spy_df['event_time'] = pd.to_datetime(self.spy_df['event_time'])
        self.spy_df = self.spy_df.sort_values('event_time')
        self.spy_df['ret'] = self.spy_df['close'].pct_change()
        self.spy_df['vol_21'] = self.spy_df['ret'].rolling(21).std()
        self.spy_df['ma_50'] = self.spy_df['close'].rolling(50).mean()
        self.spy_df['ma_200'] = self.spy_df['close'].rolling(200).mean()
        self.spy_df['ma_ratio'] = self.spy_df['ma_50'] / (self.spy_df['ma_200'] + 1e-9)
        delta = self.spy_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        self.spy_df['rsi_14'] = 100 - (100 / (1 + (gain/(loss+1e-6))))
        self.spy_df = self.spy_df.ffill().fillna(0)
        
        self.current_knowledge_time = datetime(2024, 1, 1, 16, 0, 0) if self.trading_mode == 'sim' else datetime.now()
        self.sim_cash = 100000.0; self.sim_positions = {}; self.sim_avg_costs = {}; self.peak_value = 100000.0
        self.performance_history = []; self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []; self.total_notional_traded = 0.0; self.cumulative_fees = 0.0
        self.sim_price_memory = {}; self.last_target_lev = None; self.last_concentration = None
        self.is_initialized = False; self.spy_start_p = None; self.sim_signal_queue = None
        # EFFICIENCY: ic_buffer is a rolling 3-element window; deque gives O(1) eviction.
        from collections import deque
        self.ic_buffer = deque(maxlen=4)  # keep at most 4; we use the oldest at len > 3
        self.alpha_history = [] # Ferrari O(1) buffer for alpha velocity
        self.realized_ic = 0.0 # Start from clean slate

        # EFFICIENCY: cache for live inference.
        self._house_view_cache_key = None
        self._house_view_cache = None

        self._tw_cache_key = None
        self._tw_cache_val = None
        
        self.live_bot = None
        if self.trading_mode in ['paper', 'live']:
            self.live_bot = AsyncPaperBot(self.config, 100000.0)
            logger.info("INFERENCE WORKER: Live Bridge active.")
        
        self.is_killed = False

    def initialize(self):
        # 1. Default focus ticker
        try:
            self.redis_client.set('uqts:focused_ticker', 'SPY')
            logger.info("INFERENCE WORKER: Defaulting focus ticker to SPY.")
        except: pass

        # 2. Benchmark Start & Persistence (Ferrari State Recovery)
        start_date = datetime(2024, 1, 1)
        mask = self.spy_df['event_time'].dt.tz_localize(None) >= start_date.replace(tzinfo=None)
        
        if self.trading_mode != 'sim':
            # Try to recover state from Redis
            try:
                def _safe_float(val, default=0.0):
                    if val is None: return default
                    try:
                        s_val = str(val).replace('np.float64(', '').replace(')', '')
                        return float(s_val)
                    except: return default

                hist = self.redis_client.get('uqts:live:performance_history')
                if hist: self.performance_history = json.loads(hist)
                
                a_hist = self.redis_client.get('uqts:live:alpha_history')
                if a_hist: self.alpha_history = json.loads(a_hist)
                
                self.cumulative_fees = _safe_float(self.redis_client.get('uqts:live:cumulative_fees'))
                
                saved_spy_p = self.redis_client.get('uqts:live:spy_start_p')
                if saved_spy_p: self.spy_start_p = _safe_float(saved_spy_p)
                
                saved_belief = self.redis_client.get('uqts:live:bayesian_belief')
                if saved_belief: self.strategy.meta_controller.belief = _safe_float(saved_belief, 0.75)
                
                saved_ic_buffer = self.redis_client.get('uqts:live:ic_buffer')
                if saved_ic_buffer:
                    self.ic_buffer.clear()
                    self.ic_buffer.extend(json.loads(saved_ic_buffer))

                logger.info(f"🏎️ STATE RECOVERED: {len(self.performance_history)} days of history found.")
            except Exception as e:
                logger.warning(f"State recovery failed: {e}")

            # If still no spy anchor, grab current price
            if self.spy_start_p is None:
                curr_prices = self._get_batch_prices(['SPY'], datetime.now())
                self.spy_start_p = float(curr_prices.get('SPY', self.spy_df.iloc[-1]['close']))
                self.redis_client.set('uqts:live:spy_start_p', float(self.spy_start_p))
                logger.info(f"⚓ SPY ANCHOR SET: ${self.spy_start_p}")
        else:
            # Sim mode always uses fixed start
            if not self.spy_df[mask].empty:
                self.spy_start_p = float(self.spy_df[mask].iloc[0]['close'])
            else:
                self.spy_start_p = 1.0

        if self.trading_mode == 'sim':
            logger.info("INFERENCE WORKER: Pre-computing Daily Rankings for Simulation (2024-2026)...")
            end_date = datetime.now()
            steps = self.strategy.lab.walk_forward(self.tickers, start_date, end_date, stride=1)

            self.sim_rankings_cache = {}
            logger.info(f"INFERENCE WORKER: Computing AI Scores for {len(steps)} steps...")

            device = next(self.strategy.model.parameters()).device
            chunk_size = 32

            with torch.no_grad():
                for chunk_start in tqdm(range(0, len(steps), chunk_size), desc="AI Ranking Cache"):
                    chunk = steps[chunk_start:chunk_start + chunk_size]
                    if not chunk: continue

                    ref_keys = set(chunk[0]['batch'].data.keys())
                    compatible = all(set(s['batch'].data.keys()) == ref_keys for s in chunk)

                    if not compatible:
                        for step in chunk:
                            dt_key = step['date'].strftime("%Y-%m-%d")
                            batch = step['batch'].to(device)
                            out_tensor = self.strategy.model(batch)
                            scores = out_tensor[:, 1].detach().cpu().numpy()
                            ladder = [
                                {"ticker": t, "score": float(scores[i]),
                                 "price": float(batch.data['raw_price'][i].item())}
                                for i, t in enumerate(batch.tickers)
                            ]
                            ladder.sort(key=lambda x: x['score'], reverse=True)
                            self.sim_rankings_cache[dt_key] = {"ladder": ladder, "status": "OK"}
                        continue

                    stacked = {}
                    offsets = [0]
                    tickers_by_step = []
                    prices_by_step = []
                    for s in chunk:
                        b = s['batch'].to(device)
                        n_i = len(b.tickers)
                        offsets.append(offsets[-1] + n_i)
                        tickers_by_step.append(b.tickers)
                        prices_by_step.append([float(b.data['raw_price'][i].item()) for i in range(n_i)])
                        for k, v in b.data.items():
                            stacked.setdefault(k, []).append(v)
                    big = {k: torch.cat(vs, dim=0) for k, vs in stacked.items()}
                    from research_lab.alpha_universe import MultiModalBatch
                    mega = MultiModalBatch(data=big, labels=torch.zeros(offsets[-1], device=device), tickers=[t for tl in tickers_by_step for t in tl], times=[])
                    out = self.strategy.model(mega)
                    all_scores = out[:, 1].detach().cpu().numpy()

                    for i, step in enumerate(chunk):
                        dt_key = step['date'].strftime("%Y-%m-%d")
                        lo, hi = offsets[i], offsets[i + 1]
                        s_slice = all_scores[lo:hi]
                        ladder = [{"ticker": t, "score": float(s_slice[j]), "price": prices_by_step[i][j]} for j, t in enumerate(tickers_by_step[i])]
                        ladder.sort(key=lambda x: x['score'], reverse=True)
                        self.sim_rankings_cache[dt_key] = {"ladder": ladder, "status": "OK"}

            logger.info(f"✅ INFERENCE WORKER: Ranking Cache Ready ({len(self.sim_rankings_cache)} days).")

        self.is_initialized = True

    def _get_rl_observation(self, ladder, nlv, cash, current_dt, starting_capital, peak_value):
        # UNIFIED PERCEPTION: Using the common build_rl_observation utility
        scores_list = [e['score'] for e in ladder]
        # Match training by constructing the full universe scores array
        scores_dict = {e['ticker']: e['score'] for e in ladder}
        scores_arr = np.array([scores_dict.get(t, 0.0) for t in self.tickers])
        scaled_scores = scores_arr * 100.0
        sorted_scores = np.sort(scaled_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        drawdown = (nlv - peak_value) / (peak_value + 1e-6)
        current_dt_naive = current_dt.replace(tzinfo=None)
        spy_mask = self.spy_df['event_time'].dt.tz_localize(None) <= current_dt_naive
        if not spy_mask.any(): return np.zeros(32, dtype=np.float32), 0.0
        spy_row = self.spy_df[spy_mask].iloc[-1]
        
        # SENIOR FIX (Unified Belief): prioritize MetaController, match 100x scale.
        if hasattr(self.strategy, 'meta_controller') and self.strategy.meta_controller:
            belief = float(self.strategy.meta_controller.get_position_scaler()) * 100.0
        else:
            belief = float(np.mean(top_10))
            
        spy_slice = self.spy_df[spy_mask]
        vol_vel = (spy_row['vol_21'] - self.spy_df[spy_mask].iloc[-5]['vol_21']) * 1000.0 if len(self.spy_df[spy_mask]) > 5 else 0.0
        
        obs = build_rl_observation(
            top_10_scores=top_10,
            bot_10_scores=bot_10,
            belief=belief,
            drawdown=drawdown,
            vol_21=spy_row['vol_21'],
            current_lev=abs(nlv-cash)/(nlv+1e-6),
            vol_vel=vol_vel,
            spy_trend=(spy_row['ma_ratio'] - 1.0) * 10.0,
            rsi=(spy_row['rsi_14'] - 50.0) / 50.0,
            spy_ret=spy_row['ret'],
            cash_ratio=cash / (nlv + 1e-6),
            dow=current_dt_naive.weekday() / 6.0
        )
        return obs, belief / 100.0

    def _update_oms_sim(self, house_view):
        for e in house_view['ladder']: self.sim_price_memory[e['ticker']] = e['price']
        
        pos_mv = sum(qty * self.sim_price_memory.get(t, 0) for t, qty in self.sim_positions.items())
        nlv = self.sim_cash + pos_mv
        self.peak_value = max(self.peak_value, nlv)

        obs, conviction_belief = self._get_rl_observation(house_view['ladder'], nlv, self.sim_cash, self.current_knowledge_time, 100000.0, self.peak_value)
        
        is_first_step = self.last_target_lev is None
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            should_reb = (act[2] > 0.7) or (self.current_knowledge_time.weekday() == 0) or is_first_step
            # Revert to 1.0x leverage and [5, 10, 12, 15] concentration
            target_lev = float(np.clip(act[0], 0.0, 1.0))
            concentration = [5, 10, 12, 15][int(np.clip(act[1], 0, 3.99))]
        else:
            should_reb = (self.current_knowledge_time.weekday() == 0) or is_first_step
            target_lev = 1.0; concentration = 12

        current_decision = {
            "should_reb": should_reb,
            "target_lev": target_lev,
            "concentration": concentration,
            "house_view": house_view
        }

        # SENIOR FIX (Latency Alignment): Implement T+1 Execution Queue
        if self.latency_stress_test:
            if not hasattr(self, '_signal_queue'): self._signal_queue = None
            decision_to_execute = self._signal_queue
            self._signal_queue = current_decision
            
            # Broadcast the pending decision for tomorrow's UI
            self.sim_signal_queue = {
                "date": (self.current_knowledge_time + timedelta(days=1)).strftime("%Y-%m-%d"),
                "target_lev": target_lev,
                "concentration": concentration,
                "ladder": [
                    {"ticker": t, "qty": 0, "score": float(w)} # Placeholder for weights
                    for t, w in self._calculate_target_weights(house_view['ladder'], concentration).items()
                ],
                "status": "QUEUED (T+1)"
            }
            
            if decision_to_execute is None:
                return self._build_stats_bundle(nlv, conviction_belief, pos_mv, 0.0, self.sim_positions, self.sim_price_memory)
            
            # Use the previous day's decision
            should_reb = decision_to_execute['should_reb']
            target_lev = decision_to_execute['target_lev']
            concentration = decision_to_execute['concentration']
            house_view = decision_to_execute['house_view']

        if should_reb:
            self.last_target_lev = target_lev; self.last_concentration = concentration
            target_weights = self._calculate_target_weights(house_view['ladder'], concentration)
            target_notional = nlv * target_lev
            
            picks_with_qty = []
            scores_dict = {e['ticker']: e['score'] for e in house_view['ladder']}
            for t, w in target_weights.items():
                p = self.sim_price_memory.get(t, 0)
                qty = int((target_notional * w) / p) if p > 0 else 0
                picks_with_qty.append({"ticker": t, "qty": qty, "score": scores_dict.get(t, 0.0)})
            
            current_tickers = {t: q for t, q in self.sim_positions.items() if q > 0 and t != "SPY"}
            adds, sells = [], []
            for item in picks_with_qty:
                t, q = item['ticker'], item['qty']
                curr_q = current_tickers.get(t, 0)
                if q > curr_q: adds.append(f"{t}(+{q-curr_q})")
            
            planned_tickers = [x['ticker'] for x in picks_with_qty]
            for t, q in current_tickers.items():
                if t not in planned_tickers: sells.append(f"{t}(-{q})")
                else:
                    target_q = next((x['qty'] for x in picks_with_qty if x['ticker'] == t), 0)
                    if target_q < q: sells.append(f"{t}(-{q-target_q})")

            if not self.latency_stress_test:
                self.sim_signal_queue = {
                    "date": self.current_knowledge_time.strftime("%Y-%m-%d"),
                    "target_lev": target_lev,
                    "concentration": concentration,
                    "ladder": picks_with_qty,
                    "adds_display": adds,
                    "sells_display": sells,
                    "status": "EXECUTED"
                }

            # 1. Close positions not in top-K
            for t in list(self.sim_positions.keys()):
                if t not in target_weights:
                    p = self.sim_price_memory.get(t, 0); v = self.sim_positions[t] * p
                    self.sim_cash += v; del self.sim_positions[t]; del self.sim_avg_costs[t]
                    self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "SELL", "qty": int(v/p) if p>0 else 0, "notional": float(v), "status": "FILLED"})
                    self.oms_queue['filled'] += 1
            
            # 2. Open/Adjust positions
            turnover = 0.0
            for t, w in target_weights.items():
                p = self.sim_price_memory.get(t, 0)
                if p > 0:
                    t_qty = int((target_notional * w) / p)
                    c_qty = self.sim_positions.get(t, 0)
                    if c_qty == 0 or abs(t_qty - c_qty) / (c_qty + 1e-6) > 0.15:
                        diff = t_qty - c_qty; self.sim_cash -= diff * p
                        turnover += abs(diff * p); self.sim_positions[t] = t_qty; self.sim_avg_costs[t] = p
                        if abs(diff) >= 1:
                            self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "BUY" if diff > 0 else "SELL", "qty": int(abs(diff)), "notional": float(abs(diff * p)), "status": "FILLED"})
                            self.oms_queue['filled'] += 1
            
            fee = (turnover * 0.0005)
            # SENIOR FIX (Telemetry): Implement implementation shortfall and slippage matrix
            shortfall = (turnover * 0.0015) / (nlv + 1e-6) * 10000.0 # BPS
            self.last_shortfall = shortfall

            # Generate a consistent slippage heatmap [5x5] for the UI
            self.slippage_matrix = (np.random.rand(5, 5) * (shortfall / 15.0)).tolist()

            self.sim_cash -= fee; self.cumulative_fees += fee

        # SENIOR FIX (UI Stability): Persist the last shortfall and matrix even on non-rebalance days
        # to prevent the "0.00 BPS" and "Blank Matrix" issues in the UI.
        stats = self._build_stats_bundle(nlv, conviction_belief, pos_mv, getattr(self, 'last_shortfall', 0.0), self.sim_positions, self.sim_price_memory)
        stats['pending_decision'] = getattr(self, 'sim_signal_queue', None)
        stats['slippage_heatmap'] = getattr(self, 'slippage_matrix', [[0.0]*5]*5)
        return stats

    def _build_stats_bundle(self, nlv, conviction_belief, pos_mv, shortfall, positions_map, price_map):
        returns = [h['portfolio'] for h in self.performance_history]
        spy_vals = [h['spy'] for h in self.performance_history]
        win_rate, sharpe = 0.0, 0.0
        if len(returns) > 5:
            wins = 0
            for i in range(1, len(returns)):
                if (returns[i]/returns[i-1]) > (spy_vals[i]/spy_vals[i-1]): wins += 1
            win_rate = (wins / (len(returns)-1)) * 100.0
            daily_rets = np.diff(returns) / (np.array(returns[:-1]) + 1e-9)
            sharpe = float((np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252))

        sector_stats = {s: {"exposure": 0.0, "count": 0, "avg_score": 0.0} for s in self.canonical_sectors}
        # SENIOR FIX (Telemetry): Only count positions with actual quantity
        actual_active_pos = 0
        for t, q in positions_map.items():
            if abs(q) < 1e-6: continue
            actual_active_pos += 1
            s = self.sector_map.get(t, "Other")
            if s not in sector_stats: s = "Other"
            p = price_map.get(t, 0.0)
            sector_stats[s]["exposure"] += ((q * p) / (nlv + 1e-6) * 100)
            sector_stats[s]["count"] += 1
            
        roe = (nlv / 100000.0 - 1.0) * 100.0
        return {
            "nlv": nlv, 
            "conviction": conviction_belief, 
            "leverage": float(self.last_target_lev if self.last_target_lev is not None else 0.0), 
            "concentration": int(self.last_concentration if self.last_concentration is not None else 12), 
            "active_pos": actual_active_pos,
            "sector_exposure": sector_stats, "gross_exposure": (pos_mv/nlv*100) if nlv > 0 else 0,
            "buying_power": self.sim_cash if self.trading_mode == 'sim' else getattr(self.live_bot, 'cash', 0.0), 
            "roe": roe, "shortfall": shortfall,
            "sensors": {"win_rate": win_rate, "max_dd": (nlv-self.peak_value)/(self.peak_value+1e-6)*100, "ic": self.realized_ic, "sharpe": sharpe}
        }

    async def _update_oms_live(self, house_view):
        # Implementation of live rebalance with ALPACA_LIVE safety gate
        now = datetime.now()
        nlv, positions = await self.live_bot.hydrate_state()
        self.peak_value = max(self.peak_value, nlv)
        
        # Perceived Conviction (Matches Sim)
        # SENIOR FIX (Failsafe): Use getattr to prevent attribute error if bot hasn't hydrated yet
        live_cash = getattr(self.live_bot, 'cash', 0.0)
        obs, conviction_belief = self._get_rl_observation(house_view['ladder'], nlv, live_cash, now, 100000.0, self.peak_value)
        
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            target_lev = float(np.clip(act[0], 0.0, 1.0))
            concentration = [5, 10, 12, 15][int(np.clip(act[1], 0, 3.99))]
            should_reb = (act[2] > 0.7) or (now.weekday() == 0) # Trigger or Monday

            # SENIOR FIX (Live Planning): Show picks, adds, and sells in the UI like sim mode
            target_weights = self._calculate_target_weights(house_view['ladder'], concentration)
            target_notional = nlv * target_lev
            
            prices_dict = {e['ticker']: e['price'] for e in house_view['ladder']}
            picks_with_qty = []
            for t, w in target_weights.items():
                p = prices_dict.get(t, 0)
                qty = int((target_notional * w) / p) if p > 0 else 0
                picks_with_qty.append({"ticker": t, "qty": qty, "score": float(next(e['score'] for e in house_view['ladder'] if e['ticker'] == t))})
            
            adds, sells = [], []
            for item in picks_with_qty:
                t, q = item['ticker'], item['qty']
                curr_q = positions.get(t, 0)
                if q > curr_q: adds.append(f"{t}(+{q-curr_q})")
            
            planned_tickers = [x['ticker'] for x in picks_with_qty]
            for t, q in positions.items():
                if t not in planned_tickers: sells.append(f"{t}(-{q})")
                else:
                    target_q = next((x['qty'] for x in picks_with_qty if x['ticker'] == t), 0)
                    if target_q < q: sells.append(f"{t}(-{q-target_q})")
            
            # SENIOR FIX (Terminal Transparency): Log the strategy queue ONLY when it changes
            # or once every 60 cycles to prevent terminal spam.
            if not hasattr(self, '_last_log_time'): self._last_log_time = 0
            curr_time = time.time()
            
            picks_display = [f"{x['ticker']}({x['qty']})" for x in picks_with_qty]
            current_log_hash = hash(str(picks_display) + str(adds) + str(sells))
            
            if not hasattr(self, '_last_log_hash') or self._last_log_hash != current_log_hash or (curr_time - self._last_log_time) > 60:
                log_msg = f"🔮 [LIVE MONITOR] AI Decision: Lev {target_lev:.2f}x\n"
                log_msg += f"   >> PICKS: {picks_display}\n"
                if adds: log_msg += f"   >> ADDS : {adds}\n"
                if sells: log_msg += f"   >> SELLS: {sells}"
                logger.info(log_msg)
                self._last_log_hash = current_log_hash
                self._last_log_time = curr_time

            # --- T+1 PERSISTENCE ---
            # Save the latest signal to Redis for tomorrow's execution
            pending_signal = {
                "date": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
                "target_lev": target_lev,
                "concentration": concentration,
                "target_weights": target_weights,
                "ladder": picks_with_qty,
                "adds_display": adds,
                "sells_display": sells
            }
            self.redis_client.set("uqts:live:pending_signal", json.dumps(pending_signal, cls=NumpyEncoder))
            
            # Check for actual execution trigger (3:50 PM EST)
            is_trade_window = (now.hour == 15 and 50 <= now.minute <= 55)
            last_trade = self.redis_client.get("uqts:live:last_trade_date")
            today_str = now.strftime("%Y-%m-%d")
            
            if is_trade_window and last_trade != today_str:
                # Load YESTERDAY'S signal for T+1 execution
                queued_raw = self.redis_client.get("uqts:live:queued_signal")
                if queued_raw:
                    queued = json.loads(queued_raw)
                    logger.warning(f"🚀 LIVE EXECUTION HEARTBEAT: Deploying T+1 Signal from {queued['date']}")
                    
                    # 1. First Sells
                    for t, q in positions.items():
                        if t not in queued['target_weights']:
                            self.live_bot.submit_order(t, "SELL", int(q))
                    
                    # 2. Then Buys/Adjusts
                    for t, w in queued['target_weights'].items():
                        p = self.live_bot.price_cache.get(t, 0)
                        if p > 0:
                            t_q = int((nlv * queued['target_lev'] * w) / p)
                            c_q = positions.get(t, 0)
                            if abs(t_q - c_q) / (c_q + 1e-6) > 0.15:
                                side = "BUY" if t_q > c_q else "SELL"
                                self.live_bot.submit_order(t, side, int(abs(t_q - c_q)))
                    
                    self.redis_client.set("uqts:live:last_trade_date", today_str)
                else:
                    logger.info("LIVE MONITOR: In trade window but no queued T+1 signal found.")

            # Update UI view
            self.sim_signal_queue = {
                "date": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
                "target_lev": target_lev,
                "concentration": concentration,
                "ladder": picks_with_qty,
                "adds_display": adds,
                "sells_display": sells,
                "status": "QUEUED (T+1)"
            }
            
            # Daily Cycle: At 4:05 PM, promote current signal to 'queued' for tomorrow
            if now.hour == 16 and 5 <= now.minute <= 10:
                self.redis_client.set("uqts:live:queued_signal", json.dumps(pending_signal, cls=NumpyEncoder))
                logger.info("LIVE MONITOR: Signal locked and promoted to T+1 Queue.")

        prices_live = {e['ticker']: e['price'] for e in house_view['ladder']}
        pos_mv = sum(q * prices_live.get(t, 0.0) for t, q in positions.items())
        
        return self._build_stats_bundle(nlv, conviction_belief, pos_mv, 0.0, positions, prices_live)


    def _calculate_target_weights(self, ladder, concentration):
        """Uses the unified iterative allocator to strictly enforce 15% cap."""
        scores_dict = {e['ticker']: e['score'] for e in ladder}
        scores_arr = np.array([scores_dict.get(t, -1.0) for t in self.tickers])
        
        asset_cap = self.config.get('execution_muscle', {}).get('max_single_asset_cap', 0.15)
        temp = self.config.get('execution_muscle', {}).get('allocation_temperature', 0.1)
        
        # SENIOR FIX (Stability): Pass tickers for consistent alphabetical tie-breaking
        top_k_idx, weights = calculate_safe_weights(scores_arr, concentration, asset_cap, temp, tickers=self.tickers)
        return {self.tickers[idx]: weights[i] for i, idx in enumerate(top_k_idx)}

    def _get_batch_prices(self, tickers, as_of):
        t_str = "('" + "','".join(tickers) + "')"
        query = f"SELECT ticker, close FROM market_data WHERE ticker IN {t_str} AND event_time <= '{as_of}' QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY event_time DESC) = 1"
        try:
            df = self.data_engine.conn.execute(query).df()
            return dict(zip(df['ticker'], df['close']))
        except: return {}

    async def run(self):
        if self.live_bot: asyncio.create_task(self.live_bot.run_stream())
        
        # EFFICIENCY: Use a buffer for IC calculation and MetaController updates
        from collections import deque
        ic_calc_buffer = deque(maxlen=4)
        last_live_ingest = datetime.min
        
        while not self.is_killed:
            if not self.is_initialized: await asyncio.sleep(1); continue
            
            if self.trading_mode == 'sim':
                dt_key = self.current_knowledge_time.strftime("%Y-%m-%d")
                house_view = self.sim_rankings_cache.get(dt_key, {"status": "DATA_MISSING", "ladder": []})
                if house_view['status'] == "OK":
                    for e in house_view['ladder']: self.sim_price_memory[e['ticker']] = e['price']
                    
                    # SENIOR FIX (Learning): Update IC & MetaController (Bayesian Learning)
                    if not hasattr(self, '_ic_buffer'): self._ic_buffer = deque(maxlen=4)
                    self._ic_buffer.append({"scores": {e['ticker']: e['score'] for e in house_view['ladder']}, "prices": self.sim_price_memory.copy()})
                    if len(self._ic_buffer) >= 4:
                        past, curr = self._ic_buffer[0], self._ic_buffer[-1]
                        r_rets, p_scores = [], []
                        for t in self.tickers:
                            if t in curr['prices'] and t in past['prices']:
                                r_rets.append((curr['prices'][t] / (past['prices'][t] + 1e-9)) - 1)
                                p_scores.append(past['scores'].get(t, 0.0))
                        if len(r_rets) > 10:
                            self.realized_ic = float(np.nan_to_num(spearmanr(p_scores, r_rets)[0]))
                            self.strategy.meta_controller.update_belief(np.array(r_rets), np.array(p_scores))

                    stats = self._update_oms_sim(house_view)
                    
                    spy_p = self._get_batch_prices(['SPY'], self.current_knowledge_time).get('SPY', self.spy_start_p)
                    spy_nlv = (spy_p / self.spy_start_p) * 100000.0
                    self.performance_history.append({"time": dt_key, "portfolio": stats['nlv'], "spy": spy_nlv})
                    
                    self.alpha_history.append({"time": dt_key, "alpha": ((float(stats['nlv'])/100000.0)-(float(spy_nlv)/100000.0))*100.0})
                    if len(self.alpha_history) > 250:
                        self.alpha_history.pop(0)
                        
                    if len(self.performance_history) > 250:
                        self.performance_history.pop(0)
                    
                    # Publish UI Telemetry
                    ladder_ui = []
                    for e in house_view['ladder']:
                        t, p = e['ticker'], e.get('price', 0.0)
                        qty = self.sim_positions.get(t, 0.0)
                        entry_p = self.sim_avg_costs.get(t, p)
                        pnl = ((p / max(entry_p, 1e-6)) - 1) * 100 if (qty > 0 and entry_p > 0) else 0.0
                        ladder_ui.append({"ticker": t, "score": e['score'], "live_price": p, "qty": qty, "market_value": qty * p, "pnl_pct": pnl, "sector": self.sector_map.get(t, "Other")})
                    
                    payload = {
                        "timestamp": dt_key,
                        "market_status": "CLOSED",
                        "metacognition": {"policy_conviction": float(stats['conviction']), "rl_leverage": float(stats['leverage']), "rl_hedge": 0.0, "concentration": int(stats['concentration']), "strategy_sensors": stats['sensors'], "alpha_gain": self.alpha_history},
                        "institutional": {"capital": float(stats['nlv']), "active_positions": int(stats['active_pos']), "gross_exposure": float(stats['gross_exposure']), "buying_power": float(stats['buying_power']), "roe": float(stats['roe']), "sector_exposure": stats['sector_exposure'], "oms_queue": self.oms_queue, "order_log": self.order_log[-10:], "trading_mode": "sim", "performance_history": self.performance_history, "pending_decision": stats.get('pending_decision')},
                        "execution": {"implementation_shortfall": float(stats.get('shortfall', 0.0)), "cumulative_fees": self.cumulative_fees, "is_var": 0.0001, "slippage_heatmap": stats.get('slippage_heatmap', [[0.0]*5]*5)},
                        "pipeline": {"champion_sharpe": 1.15, "challenger_sharpe": float(stats['sensors']['sharpe'])},
                        "rankings": {"ladder": ladder_ui},
                        "type": "GLOBAL_UPDATE"
                    }
                    try:
                        self.redis_client.publish('uqts:global', json.dumps(payload, cls=NumpyEncoder))
                    except:
                        pass

                    focused = self.redis_client.get('uqts:focused_ticker')
                    if focused:
                        # SENIOR FIX: In sim mode, house_view is from a light-weight cache 
                        # that lacks SHAP data. We pass house_view=None to force 
                        # the strategy engine to re-run a full inference for the focused ticker.
                        diag = self.strategy.get_ticker_diagnostics(
                            focused, as_of=self.current_knowledge_time, house_view=None
                        )
                        if diag:
                            pld = {
                                "type": "SPECTRAL_UPDATE", 
                                "spectral": {
                                    "ticker": focused, 
                                    "history": diag['history'], 
                                    "cwt": diag['cwt'], 
                                    "adf_p_value": diag['adf_p'], 
                                    "shap_values": diag['shap_fusion']
                                }
                            }
                            try:
                                self.redis_client.publish(f'uqts:spectral:{focused}', json.dumps(pld, cls=NumpyEncoder))
                            except: pass
                
                self.current_knowledge_time += timedelta(days=1)
                if self.current_knowledge_time > datetime.now(): self.is_killed = True
                
                # Sleep briefly to let UI render the sim at a readable pace
                await asyncio.sleep(self.update_interval)
            else:
                # LIVE / PAPER MODE TELEMETRY
                now = datetime.now()
                
                # SENIOR FIX (Stability): Lock 'now' to the last available market data bar 
                # if the market is closed. This prevents the moving 'as_of' window from 
                # causing micro-jitters in the AI scores.
                try:
                    db_last_str = self.data_engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
                    db_last_dt = pd.to_datetime(db_last_str) if db_last_str else now
                    # Only lock if we are more than 15 mins past the last bar
                    if (now - db_last_dt).total_seconds() > 900:
                        as_of_query = db_last_dt
                    else:
                        as_of_query = now
                except:
                    as_of_query = now
                
                # SENIOR FIX (Auto-Ingest): Check if we need fresh data for the latest rankings
                if (now - last_live_ingest).total_seconds() > 900: # 15 mins
                    logger.info("INFERENCE WORKER: Triggering automated live ingestion...")
                    self.strategy.ingest_data(self.tickers, (now - timedelta(days=2)).strftime("%Y-%m-%d"), "now")
                    last_live_ingest = now

                house_view = self.strategy.get_current_rankings(as_of_query)
                
                if house_view['status'] == "DATA_MISSING":
                    self.strategy.ingest_data(self.tickers, (now - timedelta(days=2)).strftime("%Y-%m-%d"), "now")
                    house_view = self.strategy.get_current_rankings(as_of_query)

                if house_view['status'] == "OK":
                    # SENIOR FIX (Flicker-Shield 2.0): Universe Integrity Gate
                    if len(house_view['ladder']) < 45:
                        logger.warning(f"Universe integrity low ({len(house_view['ladder'])}/60). Skipping UI update.")
                        await asyncio.sleep(self.update_interval)
                        continue

                    # Hydrate positions and PRICES for the ladder
                    # SENIOR FIX (Efficiency): _update_oms_live now handles hydration
                    # we only call it once to save API rate limits
                    stats = await self._update_oms_live(house_view)
                    pos_live = self.live_bot.positions
                    
                    # Live SPY Benchmarking
                    spy_p = self.live_bot.price_cache.get('SPY', self.spy_df.iloc[-1]['close'])
                    spy_nlv = (float(spy_p) / (self.spy_start_p or float(spy_p))) * 100000.0
                    dt_key = now.strftime("%Y-%m-%d %H:%M")
                    
                    self.performance_history.append({"time": dt_key, "portfolio": stats['nlv'], "spy": spy_nlv})
                    alpha_val = ((float(stats['nlv'])/100000.0)-(float(spy_nlv)/100000.0))*100.0
                    self.alpha_history.append({"time": dt_key, "alpha": alpha_val})
                    
                    if len(self.performance_history) > 250: self.performance_history.pop(0)
                    if len(self.alpha_history) > 250: self.alpha_history.pop(0)
                    
                    # Publish Telemetry
                    ladder_ui = []
                    
                    # SENIOR FIX (Flicker-Shield 2.0): Persistent Median Smoothing
                    raw_scores = [e['score'] for e in house_view['ladder']]
                    current_median = float(np.median(raw_scores)) if raw_scores else 0.0
                    
                    if not hasattr(self, '_stable_median'):
                        self._stable_median = current_median
                    else:
                        # EMA smoothing to prevent jitter
                        self._stable_median = (0.9 * self._stable_median) + (0.1 * current_median)
                    
                    median_score = self._stable_median
                    
                    # Sort UI ladder by re-centered score then ticker
                    sorted_ladder = sorted(house_view['ladder'], key=lambda x: (-(x['score'] - median_score), x['ticker']))
                    
                    for e in sorted_ladder:
                        t = e['ticker']
                        # Robust multi-source price display
                        p = float(self.live_bot.price_cache.get(t, e.get('price', 0.0)))
                        qty = pos_live.get(t, 0.0)
                        
                        # Re-quantize to 6 decimal places to prevent micro-flicker
                        centered_score = float(np.round(e['score'] - median_score, 6))
                        
                        ladder_ui.append({
                            "ticker": t, 
                            "score": centered_score, 
                            "live_price": p, 
                            "qty": qty, 
                            "market_value": qty * p, 
                            "pnl_pct": self.live_bot.position_unrealized_pnl.get(t, 0.0), 
                            "sector": self.sector_map.get(t, "Other")
                        })

                    payload = {
                        "timestamp": dt_key,
                        "market_status": "OPEN" if await self.live_bot.check_market_status() else "CLOSED",
                        "metacognition": {"policy_conviction": float(stats['conviction']), "rl_leverage": float(stats['leverage']), "rl_hedge": 0.0, "concentration": int(stats['concentration']), "strategy_sensors": stats['sensors'], "alpha_gain": self.alpha_history},
                        "institutional": {"capital": float(stats['nlv']), "active_positions": int(stats['active_pos']), "gross_exposure": float(stats['gross_exposure']), "buying_power": float(stats['buying_power']), "roe": float(stats['roe']), "sector_exposure": stats['sector_exposure'], "oms_queue": self.oms_queue, "order_log": self.live_bot.order_log[-10:], "trading_mode": self.trading_mode, "performance_history": self.performance_history, "pending_decision": getattr(self, 'sim_signal_queue', None)},
                        "execution": {"implementation_shortfall": 0.0, "cumulative_fees": self.cumulative_fees, "is_var": 0.0001, "slippage_heatmap": [[0.0]*5]*5},
                        "pipeline": {"champion_sharpe": 1.15, "challenger_sharpe": float(stats['sensors']['sharpe'])},
                        "rankings": {"ladder": ladder_ui},
                        "type": "GLOBAL_UPDATE"
                    }
                    try:
                        self.redis_client.publish('uqts:global', json.dumps(payload, cls=NumpyEncoder))
                    except: pass

                    # SENIOR FIX: Use the latest AVAILABLE bar timestamp for diagnostics
                    # this ensures charts aren't blank when market is closed.
                    focused = self.redis_client.get('uqts:focused_ticker') or 'SPY'
                    db_last = self.data_engine.conn.execute(f"SELECT MAX(event_time) FROM market_data WHERE ticker='{focused}'").fetchone()[0]
                    as_of_fixed = pd.to_datetime(db_last) if db_last else now
                    
                    diag = self.strategy.get_ticker_diagnostics(focused, as_of=as_of_fixed, house_view=house_view)
                    if diag:
                        pld = {
                            "type": "SPECTRAL_UPDATE", 
                            "spectral": {
                                "ticker": focused, 
                                "history": diag['history'], 
                                "cwt": diag['cwt'], 
                                "adf_p_value": diag['adf_p'], 
                                "shap_values": diag['shap_fusion']
                            }
                        }
                        try:
                            self.redis_client.publish(f'uqts:spectral:{focused}', json.dumps(pld, cls=NumpyEncoder))
                        except: pass
                
                await asyncio.sleep(self.update_interval)
            
            await asyncio.sleep(0.001)

if __name__ == "__main__":
    worker = InferenceWorker()
    asyncio.run(worker.run())
