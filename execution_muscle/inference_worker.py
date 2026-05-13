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
    V7.4.3: Mission Control Telemetry (Final Polish)
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000) / 1000.0
        self.trading_mode = self.config.get('execution_muscle', {}).get('trading_mode', 'sim')
        self.latency_stress_test = self.config.get('execution_muscle', {}).get('latency_stress_test', False)
        
        logger.info(f"INFERENCE WORKER: Initializing Master Sniper V7.4.3 (Full Telemetry).")
        if self.trading_mode == 'sim' and self.latency_stress_test:
            logger.warning("⚠️ LAGGED EXECUTION ENABLED: Decisions made on Day T will execute on Day T+1 prices.")
        
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
            "KO": "Retail", "WMT": "Retail", "MCD": "McDonald's Corp.", "PM": "Philip Morris Int.", "LOW": "Lowe's Companies",
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
        self.ic_buffer = [] # To store (date, scores_dict, prices_at_T) for Realized IC
        self.alpha_history = [] # Ferrari O(1) buffer for alpha velocity
        self.realized_ic = 0.0 # Start from clean slate
        
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
                hist = self.redis_client.get('uqts:live:performance_history')
                if hist: self.performance_history = json.loads(hist)
                
                a_hist = self.redis_client.get('uqts:live:alpha_history')
                if a_hist: self.alpha_history = json.loads(a_hist)
                
                fees = self.redis_client.get('uqts:live:cumulative_fees')
                if fees: self.cumulative_fees = float(fees)
                
                saved_spy_p = self.redis_client.get('uqts:live:spy_start_p')
                if saved_spy_p: self.spy_start_p = float(saved_spy_p)
                
                saved_belief = self.redis_client.get('uqts:live:bayesian_belief')
                if saved_belief: self.strategy.meta_controller.belief = float(saved_belief)
                
                saved_ic_buffer = self.redis_client.get('uqts:live:ic_buffer')
                if saved_ic_buffer: self.ic_buffer = json.loads(saved_ic_buffer)

                logger.info(f"🏎️ STATE RECOVERED: {len(self.performance_history)} days of history found.")
            except Exception as e:
                logger.warning(f"State recovery failed: {e}")

            # If still no spy anchor, grab current price
            if self.spy_start_p is None:
                # Use current price if available, else latest historical
                curr_prices = self._get_batch_prices(['SPY'], datetime.now())
                self.spy_start_p = float(curr_prices.get('SPY', self.spy_df.iloc[-1]['close']))
                self.redis_client.set('uqts:live:spy_start_p', self.spy_start_p)
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
            with torch.no_grad():
                device = next(self.strategy.model.parameters()).device
                for step in tqdm(steps, desc="AI Ranking Cache"):
                    dt_key = step['date'].strftime("%Y-%m-%d")
                    batch = step['batch'].to(device)
                    out_tensor = self.strategy.model(batch)
                    scores = out_tensor[:, 1].cpu().numpy()
                    
                    ladder = []
                    for i, t in enumerate(batch.tickers):
                        ladder.append({
                            "ticker": t,
                            "score": float(scores[i]),
                            "price": float(batch.data['raw_price'][i].item())
                        })
                    ladder.sort(key=lambda x: x['score'], reverse=True)
                    self.sim_rankings_cache[dt_key] = {"ladder": ladder, "status": "OK"}
            logger.info(f"✅ INFERENCE WORKER: Ranking Cache Ready ({len(self.sim_rankings_cache)} days).")

        self.is_initialized = True

    def _is_market_open(self, dt):
        """Check if US Market is currently open (9:30 - 16:00 EST)."""
        if dt.weekday() >= 5: return False
        m_start = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        m_end = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        return m_start <= dt <= m_end

    def _get_rl_observation(self, ladder, nlv, cash, current_dt, starting_capital, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history):
        # UNIFIED PERCEPTION: Scale scores by 100x to match training and SimulationEngineV5
        scores_list = [e['score'] for e in ladder]
        scores_np = np.array(scores_list) * 100.0
        sorted_scores = np.sort(scores_np)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        drawdown = (nlv - peak_value) / (peak_value + 1e-6)
        current_dt_naive = current_dt.replace(tzinfo=None)
        spy_mask = self.spy_df['event_time'].dt.tz_localize(None) <= current_dt_naive
        if not spy_mask.any(): return np.zeros(32, dtype=np.float32), 0.0
        spy_row = self.spy_df[spy_mask].iloc[-1]
        
        belief = np.mean(top_10)
        vol = spy_row.get('vol_21', 0.02)
        
        spy_slice = self.spy_df[spy_mask]
        if len(spy_slice) > 5:
            vol_vel = (spy_row['vol_21'] - spy_slice.iloc[-5]['vol_21']) * 1000.0
        else:
            vol_vel = 0.0
            
        long_mv = nlv - cash
        current_lev = abs(long_mv) / (nlv + 1e-6)
        spy_trend = (spy_row.get('ma_ratio', 1.0) - 1.0) * 10.0
        rsi = (spy_row.get('rsi_14', 50.0) - 50.0) / 50.0
        spy_ret_yest = spy_row.get('ret', 0.0) * 10.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, vol, current_lev],
            [vol_vel, spy_trend, rsi, spy_ret_yest],
            [cash/(nlv+1e-6), 0.0, 1.0, current_dt_naive.weekday()/6.0]
        ]).astype(np.float32)
        
        conviction = np.clip(belief / 100.0, 0.0, 1.0)
        return np.clip(np.nan_to_num(obs), -10.0, 10.0), conviction

    def _update_oms_sim(self, house_view):
        if self.trading_mode != 'sim': return
        
        # Point-in-Time Price Memory
        for e in house_view['ladder']:
            self.sim_price_memory[e['ticker']] = e['price']
        
        pos_mv = sum(qty * self.sim_price_memory.get(t, 0) for t, qty in self.sim_positions.items())
        nlv = self.sim_cash + pos_mv
        self.peak_value = max(self.peak_value, nlv)

        obs, conviction_belief = self._get_rl_observation(house_view['ladder'], nlv, self.sim_cash, self.current_knowledge_time, 100000.0, self.peak_value, [], 0.0, 0.0, [])
        
        # 1. Determine Decision for TODAY (Signal Generation)
        is_first_step = self.last_target_lev is None
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            should_reb_signal = (act[2] > 0.7) or (self.current_knowledge_time.weekday() == 0) or is_first_step
            target_lev_signal = 1.0 if act[0] > 0.5 else 0.0
            concentration_signal = [5, 8, 12, 15][int(np.clip(act[1], 0, 3.99))]
        else:
            should_reb_signal = (self.current_knowledge_time.weekday() == 0) or is_first_step
            target_lev_signal = 1.0; concentration_signal = 12

        # ESTIMATE QUANTITIES FOR PLANNING
        target_notional = nlv * target_lev_signal
        target_weights = self._calculate_target_weights(house_view['ladder'], concentration_signal)
        
        picks_with_qty = []
        for t, w in target_weights.items():
            p = self.sim_price_memory.get(t, 0)
            qty = int((target_notional * w) / p) if p > 0 else 0
            picks_with_qty.append({"ticker": t, "qty": qty, "score": next(x['score'] for x in house_view['ladder'] if x['ticker'] == t)})

        current_decision = {
            "date": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M"),
            "should_reb": should_reb_signal,
            "target_lev": target_lev_signal,
            "concentration": concentration_signal,
            "ladder": picks_with_qty,
            "status": "QUEUED"
        }

        # 4. Final Telemetry Build
        bayesian_belief = self.strategy.meta_controller.get_position_scaler()
        fused_conviction = bayesian_belief * target_lev_signal
        
        # --- LAGGED VS INSTANT EXECUTION ---
        if self.latency_stress_test:
            # 1-Day Lag: Pull YESTERDAY'S decision to execute TODAY
            decision_to_execute = self.sim_signal_queue
            
            # Print planning log for NEXT day
            if should_reb_signal:
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
                        target_q = next(x['qty'] for x in picks_with_qty if x['ticker'] == t)
                        if target_q < q: sells.append(f"{t}(-{q-target_q})")

                current_decision['adds_display'] = adds
                current_decision['sells_display'] = sells

                picks_display = [f"{x['ticker']}({x['qty']})" for x in picks_with_qty]
                log_msg = f"🔮 [PLANNING] AI Decision Queued for Next Day: Lev {target_lev_signal}x\n"
                log_msg += f"   >> PICKS: {picks_display}\n"
                if adds: log_msg += f"   >> ADDS : {adds}\n"
                if sells: log_msg += f"   >> SELLS: {sells}"
                logger.info(log_msg)

            self.sim_signal_queue = current_decision
            if decision_to_execute is None:
                self.last_target_lev = 0.0; self.last_concentration = 5
                return self._build_stats_bundle(nlv, fused_conviction, pos_mv, 0.0)
        else:
            decision_to_execute = current_decision

        # 3. Execute decision
        should_reb = decision_to_execute['should_reb']
        target_lev = decision_to_execute['target_lev']
        concentration = decision_to_execute['concentration']
        ladder_to_use = decision_to_execute['ladder']

        total_shortfall = getattr(self, 'last_shortfall', 0.0)
        if should_reb:
            self.last_target_lev = target_lev
            self.last_concentration = concentration
            
            target_notional = nlv * target_lev
            target_weights = self._calculate_target_weights(ladder_to_use, concentration)
            top_picks = list(target_weights.keys())
            
            turnover_notional = 0.0
            for t in list(self.sim_positions.keys()):
                if t not in top_picks:
                    p = self.sim_price_memory.get(t, 0)
                    v = self.sim_positions[t] * p
                    self.sim_cash += v; turnover_notional += v
                    self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "SELL", "qty": int(self.sim_positions[t]), "notional": float(v), "status": "FILLED"})
                    self.oms_queue['filled'] += 1
                    del self.sim_positions[t]; del self.sim_avg_costs[t]
                    
            for t, w in target_weights.items():
                p = self.sim_price_memory.get(t, 0)
                if p > 0:
                    t_qty = int((target_notional * w) / p)
                    c_qty = self.sim_positions.get(t, 0)
                    if c_qty == 0 or abs(t_qty - c_qty) / (c_qty + 1e-6) > 0.15:
                        diff_v = (t_qty - c_qty) * p
                        self.sim_cash -= diff_v
                        turnover_notional += abs(diff_v)
                        self.sim_positions[t] = t_qty
                        self.sim_avg_costs[t] = p
                        if abs(t_qty - c_qty) >= 1:
                            self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "BUY" if t_qty > c_qty else "SELL", "qty": int(abs(t_qty-c_qty)), "notional": float(abs(diff_v)), "status": "FILLED"})
                            self.oms_queue['filled'] += 1
            
            shortfall_val = turnover_notional * (0.0005 + np.random.uniform(0, 0.0002))
            self.sim_cash -= shortfall_val
            self.cumulative_fees += shortfall_val
            total_shortfall = (shortfall_val / (nlv + 1e-6)) * 10000.0 # BPS
            self.last_shortfall = total_shortfall

        return self._build_stats_bundle(nlv, fused_conviction, pos_mv, total_shortfall)

    def _build_stats_bundle(self, nlv, conviction_belief, pos_mv, total_shortfall):
        returns = [h['portfolio'] for h in self.performance_history]
        spy_vals = [h['spy'] for h in self.performance_history]
        win_rate, sharpe = 0.0, 0.0
        if len(returns) > 5:
            wins = 0
            for i in range(1, len(returns)):
                if (returns[i]/returns[i-1]) > (spy_vals[i]/spy_vals[i-1]): wins += 1
            win_rate = (wins / (len(returns)-1)) * 100.0
            daily_rets = np.diff(returns) / (np.array(returns[:-1]) + 1e-9)
            sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252)

        sector_stats, _ = self._get_sector_exposure(self.sim_positions, self.sim_price_memory, nlv)
        active_pos_count = len([q for q in self.sim_positions.values() if q != 0])
        roe = (nlv / 100000.0 - 1.0) * 100.0
        return {
            "nlv": nlv, "conviction": conviction_belief, "leverage": getattr(self, 'last_target_lev', 0.0), "hedge": 0.0, "concentration": getattr(self, 'last_concentration', 5),
            "active_pos": active_pos_count, "sector_exposure": sector_stats, "gross_exposure": (pos_mv/nlv*100) if nlv > 0 else 0,
            "buying_power": self.sim_cash, "roe": roe, "shortfall": total_shortfall,
            "sensors": {"win_rate": win_rate, "max_dd": (nlv-self.peak_value)/(self.peak_value+1e-6)*100, "ic": self.realized_ic, "sharpe": sharpe},
            "pending_decision": self.sim_signal_queue
        }

    async def _update_oms_live(self, house_view):
        now = datetime.now()
        is_thinking_window = (now.hour == 16 and 5 <= now.minute <= 30)
        is_execution_window = (now.hour == 15 and now.minute >= 50)

        pending_json = self.redis_client.get('uqts:live:pending_decision')
        pending_decision = json.loads(pending_json) if pending_json else None
        
        today_str = now.strftime("%Y-%m-%d")
        last_plan_date = self.redis_client.get('uqts:live:last_plan_date')
        
        should_catch_up = ((now.hour >= 16 or now.hour < 15) and pending_decision is None and last_plan_date != today_str and not is_thinking_window and house_view.get('status') == "OK")
        if should_catch_up:
            logger.warning("🏎️ RETROACTIVE PLANNING: Catch-up initiated.")
            is_thinking_window = True 

        nlv, _ = await self.live_bot.hydrate_state()
        self.peak_value = max(self.peak_value, nlv)
        prices = self._get_batch_prices(self.tickers, now)
        long_mv = sum(qty * prices.get(t, 0) for t, qty in self.live_bot.positions.items())
        roe = (nlv / 100000.0 - 1.0) * 100.0

        if is_thinking_window:
            last_plan_date = self.redis_client.get('uqts:live:last_plan_date')
            today_str = now.strftime("%Y-%m-%d")
            if last_plan_date != today_str:
                obs, _ = self._get_rl_observation(house_view['ladder'], nlv, nlv - long_mv, now, 100000.0, self.peak_value, [], 0.0, 0.0, [])
                if self.rl_pilot:
                    act, _ = self.rl_pilot.predict(obs, deterministic=True)
                    target_lev = 1.0 if act[0] > 0.5 else 0.0
                    concentration = [5, 8, 12, 15][int(np.clip(act[1], 0, 3.99))]
                else:
                    target_lev = 1.0; concentration = 12

                target_notional = nlv * target_lev
                target_weights = self._calculate_target_weights(house_view['ladder'], concentration)
                picks_with_qty = []
                for t, w in target_weights.items():
                    p = prices.get(t, 0)
                    qty = int((target_notional * w) / p) if p > 0 else 0
                    picks_with_qty.append({"ticker": t, "qty": qty, "score": next(x['score'] for x in house_view['ladder'] if x['ticker'] == t)})

                current_tickers = {t: q for t, q in self.live_bot.positions.items() if q > 0 and t != "SPY"}
                adds, sells = [], []
                for item in picks_with_qty:
                    t, q = item['ticker'], item['qty']
                    curr_q = current_tickers.get(t, 0)
                    if q > curr_q: adds.append(f"{t}(+{q-curr_q})")
                
                planned_tickers = [x['ticker'] for x in picks_with_qty]
                for t, q in current_tickers.items():
                    if t not in planned_tickers: sells.append(f"{t}(-{q})")
                    else:
                        target_q = next(x['qty'] for x in picks_with_qty if x['ticker'] == t)
                        if target_q < q: sells.append(f"{t}(-{q-target_q})")

                decision = {"date": today_str, "target_lev": target_lev, "concentration": concentration, "ladder": picks_with_qty, "adds_display": adds, "sells_display": sells, "status": "QUEUED"}
                self.redis_client.set('uqts:live:pending_decision', json.dumps(decision, cls=NumpyEncoder))
                self.redis_client.set('uqts:live:last_plan_date', today_str)
                logger.success(f"🔮 [PLANNING] AI Decision Queued for tomorrow: Lev {target_lev}x")

        if is_execution_window and pending_decision:
            if pending_decision['date'] != today_str:
                logger.info(f"🚀 [EXECUTION] Routing MOC orders...")
                target_lev = pending_decision['target_lev']
                concentration = pending_decision['concentration']
                target_weights = self._calculate_target_weights(pending_decision['ladder'], concentration)
                total_target_capital = nlv * target_lev
                turnover_notional = 0.0
                for t, q in list(self.live_bot.positions.items()):
                    if t not in target_weights and t != "SPY" and q > 0:
                        turnover_notional += q * prices.get(t, 0)
                        self.live_bot.submit_order(t, "SELL", int(q))
                for t, w in target_weights.items():
                    p = prices.get(t, 0)
                    if p <= 0: continue
                    target_qty = int((total_target_capital * w) / p)
                    current_qty = self.live_bot.positions.get(t, 0)
                    diff = target_qty - current_qty
                    if abs(diff) >= 1:
                        turnover_notional += abs(diff) * p
                        self.live_bot.submit_order(t, "BUY" if diff > 0 else "SELL", int(abs(diff)))
                self.cumulative_fees += turnover_notional * 0.0005
                self.redis_client.delete('uqts:live:pending_decision')

        bayesian_belief = self.strategy.meta_controller.get_position_scaler()
        rl_leverage = 0.0
        if house_view.get('status') == "OK":
            obs, _ = self._get_rl_observation(house_view['ladder'], nlv, nlv - long_mv, now, 100000.0, self.peak_value, [], 0.0, 0.0, [])
            if self.rl_pilot:
                act, _ = self.rl_pilot.predict(obs, deterministic=True)
                rl_leverage = float(act[0])
            else: rl_leverage = 1.0
        
        fused_conviction = bayesian_belief * rl_leverage
        sector_stats, _ = self._get_sector_exposure(self.live_bot.positions, prices, nlv)
        active_pos_count = len([q for q in self.live_bot.positions.values() if q != 0])
        
        return {
            "nlv": nlv, "conviction": fused_conviction, "leverage": pending_decision['target_lev'] if pending_decision else 0.0, "hedge": 0.0, "concentration": pending_decision['concentration'] if pending_decision else 5,
            "active_pos": active_pos_count, "sector_exposure": sector_stats, "gross_exposure": (long_mv/nlv*100) if nlv > 0 else 0,
            "buying_power": nlv-long_mv, "roe": roe, "shortfall": getattr(self, 'last_shortfall_live', 0.0), "cumulative_fees": self.cumulative_fees,
            "sensors": {"win_rate": getattr(self, 'live_win_rate', 0.0), "max_dd": (nlv-self.peak_value)/(self.peak_value+1e-6)*100, "ic": self.realized_ic, "sharpe": getattr(self, 'live_sharpe', 0.0)},
            "pending_decision": pending_decision
        }

    def _get_sector_exposure(self, positions, prices, nlv):
        stats = {s: {"exposure": 0.0, "count": 0, "avg_score": 0.0} for s in self.canonical_sectors}
        total_long_mv = 0.0
        for t, q in positions.items():
            s = self.sector_map.get(t, "Other"); p = prices.get(t, 0.0); mv = q * p
            stats[s]["exposure"] += (mv / (nlv + 1e-6) * 100); stats[s]["count"] += 1; total_long_mv += mv
        return stats, total_long_mv

    def _calculate_target_weights(self, valid_ladder, concentration):
        top_entries = valid_ladder[:concentration]
        # Professional High-Density Allocation: Temperature control via config
        temp = self.config.get('execution_muscle', {}).get('allocation_temperature', 0.5)
        asset_cap = self.config.get('execution_muscle', {}).get('max_single_asset_cap', 0.15)
        
        top_scores = np.array([e['score'] for e in top_entries]) * 100.0
        exp_scores = np.exp((top_scores - np.max(top_scores)) / temp)
        weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        
        # Enforce safety cap and redistribute if needed (Institutional Safety)
        if np.max(weights) > asset_cap:
            weights = np.clip(weights, 0, asset_cap)
            weights = weights / np.sum(weights)
            
        return {top_entries[i]['ticker']: float(weights[i]) for i in range(len(top_entries))}

    def _get_batch_prices(self, tickers, as_of):
        if self.trading_mode != 'sim' and self.live_bot:
            try:
                headers = {"APCA-API-KEY-ID": self.live_bot.api_key, "APCA-API-SECRET-KEY": self.live_bot.api_secret}
                url = f"https://data.alpaca.markets/v2/stocks/trades/latest?symbols={','.join(tickers)}"
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    prices = {t: float(d['p']) for t, d in data.get('trades', {}).items()}
                    if prices: return prices
            except Exception as e:
                logger.warning(f"Live Price Fetch Failed ({e}), falling back to local DB.")
    def _get_batch_prices(self, tickers, as_of):
        if self.trading_mode != 'sim' and self.live_bot:
            try:
                headers = {"APCA-API-KEY-ID": self.live_bot.api_key, "APCA-API-SECRET-KEY": self.live_bot.api_secret}
                url = f"https://data.alpaca.markets/v2/stocks/trades/latest?symbols={','.join(tickers)}"
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    prices = {t: float(d['p']) for t, d in data.get('trades', {}).items()}
                    if prices: return prices
            except Exception as e:
                logger.warning(f"Live Price Fetch Failed ({e}), falling back to local DB.")

        t_tuple = tuple(tickers)
        t_str = f"('{t_tuple[0]}')" if len(t_tuple) == 1 else str(t_tuple)
        query = f"SELECT ticker, close FROM market_data WHERE ticker IN {t_str} AND event_time <= '{as_of.strftime('%Y-%m-%d %H:%M:%S')}' QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY event_time DESC) = 1"
        try:
            df = self.data_engine.conn.execute(query).df()
            return dict(zip(df['ticker'], df['close']))
        except: return {}

    async def run(self):
        logger.info(f"INFERENCE WORKER: MASTER SNIPER STARTING ({self.trading_mode})")
        if self.live_bot: asyncio.create_task(self.live_bot.run_stream())
        tick = 0
        while not self.is_killed:
            tick += 1
            if not self.is_initialized: await asyncio.sleep(1); continue
            if self.trading_mode != 'sim': 
                self.current_knowledge_time = datetime.now()
                house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            else:
                if self.current_knowledge_time.weekday() >= 5: self.current_knowledge_time += timedelta(days=1); continue
                dt_key = self.current_knowledge_time.strftime("%Y-%m-%d")
                house_view = self.sim_rankings_cache.get(dt_key, {"status": "DATA_MISSING", "ladder": []})

            if house_view['status'] == "OK":
                if self.trading_mode == 'sim': stats = self._update_oms_sim(house_view)
                else: stats = await self._update_oms_live(house_view)
            else:
                if self.trading_mode != 'sim': stats = await self._update_oms_live(house_view)
                else: stats = None

            if stats:
                prices = self.sim_price_memory if self.trading_mode == 'sim' else self._get_batch_prices(self.tickers, self.current_knowledge_time)
                scores_dict = {e['ticker']: e['score'] for e in house_view.get('ladder', [])}
                self.ic_buffer.append({"date": self.current_knowledge_time, "scores": scores_dict, "prices": prices.copy()})
                if len(self.ic_buffer) > 3:
                    past_data = self.ic_buffer.pop(0)
                    realized_rets, pred_scores = [], []
                    for t in self.tickers:
                        if t in prices and t in past_data['prices']:
                            realized_rets.append((prices[t]/(past_data['prices'][t]+1e-9))-1)
                            pred_scores.append(past_data['scores'].get(t, 0.0))
                    if len(realized_rets) > 10:
                        self.strategy.meta_controller.update_belief(np.array(realized_rets), np.array(pred_scores))
                        ic_val, _ = spearmanr(pred_scores, realized_rets)
                        self.realized_ic = max(0, ic_val)

                ladder_ui = []
                for t in self.tickers:
                    p = prices.get(t, 0.0); score = scores_dict.get(t, 0.0)
                    qty = self.sim_positions.get(t, 0.0) if self.trading_mode == 'sim' else self.live_bot.positions.get(t, 0.0)
                    entry_p = self.sim_avg_costs.get(t, p) if self.trading_mode == 'sim' else p
                    mv, pnl = qty * p, ((p/max(entry_p, 1e-6))-1)*100 if entry_p > 0 else 0.0
                    ladder_ui.append({"ticker": t, "score": float(score), "live_price": float(p), "qty": int(qty), "market_value": float(mv), "pnl_pct": float(pnl), "sector": self.sector_map.get(t, "Other")})

                focused = self.redis_client.get('uqts:focused_ticker')
                if focused:
                    diag = self.strategy.get_ticker_diagnostics(focused, as_of=self.current_knowledge_time)
                    if diag:
                        pld = {"type": "SPECTRAL_UPDATE", "spectral": {"ticker": focused, "history": diag['history'], "cwt": diag['cwt'], "adf_p_value": diag['adf_p'], "shap_values": diag['shap_fusion']}}
                        self.redis_client.publish(f'uqts:spectral:{focused}', json.dumps(pld, cls=NumpyEncoder))

                spy_p = prices.get('SPY', self.spy_start_p or 1.0)
                spy_cap = (spy_p / self.spy_start_p) * 100000.0 if self.spy_start_p else 100000.0
                gain_pct = (stats['nlv'] / 100000.0 - 1) * 100.0
                timestamp_str = self.current_knowledge_time.strftime("%Y-%m-%d")
                self.performance_history.append({"time": timestamp_str, "portfolio": float(stats['nlv']), "spy": float(spy_cap)})
                self.alpha_history.append({"time": timestamp_str, "alpha": ((float(stats['nlv'])/100000.0)-(float(spy_cap)/100000.0))*100.0})

                if self.trading_mode != 'sim' and tick % 10 == 0:
                    self.redis_client.set('uqts:live:performance_history', json.dumps(self.performance_history))
                    self.redis_client.set('uqts:live:alpha_history', json.dumps(self.alpha_history))
                    self.redis_client.set('uqts:live:cumulative_fees', self.cumulative_fees)
                    self.redis_client.set('uqts:live:bayesian_belief', self.strategy.meta_controller.belief)
                    self.redis_client.set('uqts:live:ic_buffer', json.dumps(self.ic_buffer, cls=NumpyEncoder))

                payload = {
                    "timestamp": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "market_status": "OPEN" if self._is_market_open(self.current_knowledge_time) else "CLOSED",
                    "metacognition": {"policy_conviction": float(stats['conviction']), "rl_leverage": float(stats['leverage']), "rl_hedge": 0.0, "concentration": int(stats['concentration']), "strategy_sensors": stats['sensors'], "alpha_gain": self.alpha_history},
                    "institutional": {"capital": float(stats['nlv']), "active_positions": int(stats['active_pos']), "gross_exposure": float(stats['gross_exposure']), "buying_power": float(stats['buying_power']), "roe": float(stats['roe']), "sector_exposure": stats['sector_exposure'], "oms_queue": self.live_bot.oms_stats if self.live_bot else self.oms_queue, "order_log": self.live_bot.order_log[-10:] if self.live_bot else self.order_log[-10:], "trading_mode": self.trading_mode, "performance_history": self.performance_history, "pending_decision": stats.get('pending_decision')},
                    "execution": {"implementation_shortfall": float(stats['shortfall']), "cumulative_fees": float(self.cumulative_fees), "is_var": 0.0001, "slippage_heatmap": [[float(min(1.0, stats['shortfall']*0.1 + np.random.random()*0.1)) for _ in range(5)] for _ in range(5)]},
                    "pipeline": {"champion_sharpe": 1.15, "challenger_sharpe": float(stats['sensors']['sharpe'])},
                    "rankings": {"ladder": ladder_ui}, "type": "GLOBAL_UPDATE"
                }
                self.redis_client.publish('uqts:global', json.dumps(payload, cls=NumpyEncoder))

            if self.trading_mode == 'sim': 
                self.current_knowledge_time += timedelta(days=1)
                if self.current_knowledge_time > datetime.now(): self.is_killed = True
            await asyncio.sleep(self.update_interval)

if __name__ == "__main__":
    worker = InferenceWorker()
    asyncio.run(worker.run())
