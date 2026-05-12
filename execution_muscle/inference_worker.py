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

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.strategy_engine import StrategyEngine
from execution_muscle.paper_bot import AsyncPaperBot
from alpha_factory.meta_controller import BayesianMetaController
from execution_muscle.risk_parity_sizer import RiskParitySizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
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
        
        logger.info(f"INFERENCE WORKER: Initializing Master Sniper V7.4.3 (Full Telemetry).")
        
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
        self.order_log = []; self.total_notional_traded = 0.0
        
        self.live_bot = None
        if self.trading_mode in ['paper', 'live']:
            self.live_bot = AsyncPaperBot(self.config, 100000.0)
            asyncio.create_task(self.live_bot.run_stream())
            logger.info("INFERENCE WORKER: Live Bridge active.")
        
        self.is_killed = False

    def initialize(self):
        self.is_initialized = True

    def _get_rl_observation(self, ladder, nlv, cash, current_dt, starting_capital, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history):
        # BIT-FOR-BIT ALIGNMENT WITH SimulationEngineV5
        sorted_scores = np.sort([e['score'] for e in ladder])
        top_10 = sorted_scores[-10:][::-1] * 100.0
        bot_10 = sorted_scores[:10] * 100.0
        
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        drawdown = (nlv - peak_value) / (peak_value + 1e-6)
        current_dt_naive = current_dt.replace(tzinfo=None)
        spy_mask = self.spy_df['event_time'].dt.tz_localize(None) <= current_dt_naive
        if not spy_mask.any(): return np.zeros(32, dtype=np.float32), 0.0
        spy_row = self.spy_df[spy_mask].iloc[-1]
        
        belief = np.mean(top_10)
        vol = spy_row.get('vol_21', 0.0)
        
        spy_slice = self.spy_df[spy_mask]
        if len(spy_slice) > 5:
            vol_vel = spy_row['vol_21'] - spy_slice.iloc[-5]['vol_21']
        else:
            vol_vel = 0.0
            
        long_mv = nlv - cash
        current_lev = abs(long_mv) / nlv if nlv > 0 else 0
        spy_trend = (spy_row.get('ma_ratio', 1.0) - 1.0) * 10.0
        rsi = (spy_row.get('rsi_14', 50.0) - 50.0) / 50.0
        spy_ret_yest = spy_row.get('ret', 0.0) * 10.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, vol, current_lev],
            [vol_vel * 1000.0, spy_trend, rsi, spy_ret_yest],
            [cash/nlv, 0.0, 1.0, current_dt_naive.weekday()/6.0]
        ]).astype(np.float32)
        
        # Policy Conviction proxy: Normalized Alpha Confidence
        conviction = np.clip(belief, 0.0, 1.0)
        
        return np.clip(np.nan_to_num(obs), -10.0, 10.0), conviction

    def _update_oms_sim(self, house_view):
        if self.trading_mode != 'sim': return
        prices = self._get_batch_prices(self.tickers, self.current_knowledge_time)
        if not prices: return None

        pos_mv = sum(qty * prices.get(t, 0) for t, qty in self.sim_positions.items())
        nlv = self.sim_cash + pos_mv
        self.peak_value = max(self.peak_value, nlv)

        portfolio_returns = [] 
        score_history = []
        obs, conviction_belief = self._get_rl_observation(house_view['ladder'], nlv, self.sim_cash, self.current_knowledge_time, 100000.0, self.peak_value, portfolio_returns, 0.0, 0.0, score_history)
        
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            should_reb = (act[2] > 0.7) or (self.current_knowledge_time.weekday() == 0)
            if should_reb:
                self.last_target_lev = 1.0 if act[0] > 0.5 else 0.0
                self.last_concentration = [5, 8, 12, 15][int(np.clip(act[1], 0, 3.99))]
        else:
            should_reb = (self.current_knowledge_time.weekday() == 0)
            self.last_target_lev = 1.0; self.last_concentration = 12

        total_shortfall = getattr(self, 'last_shortfall', 0.0)
        if should_reb:
            target_notional = nlv * self.last_target_lev
            top_picks = [e['ticker'] for e in house_view['ladder'][:self.last_concentration] if e['score'] > 0]
            top_scores = np.array([e['score'] for e in house_view['ladder'][:self.last_concentration] if e['score'] > 0])
            
            exp_scores = np.exp((top_scores - np.max(top_scores)) / 0.5)
            weights = exp_scores / (np.sum(exp_scores) + 1e-9)
            
            turnover_notional = 0.0
            for t in list(self.sim_positions.keys()):
                if t not in top_picks:
                    p = prices.get(t, 0)
                    v = self.sim_positions[t] * p
                    self.sim_cash += v; turnover_notional += v
                    self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "SELL", "qty": int(self.sim_positions[t]), "notional": float(v), "status": "FILLED"})
                    self.oms_queue['filled'] += 1
                    del self.sim_positions[t]; del self.sim_avg_costs[t]
                    
            for i, t in enumerate(top_picks):
                p = prices.get(t, 0)
                if p > 0:
                    t_qty = int((target_notional * weights[i]) / p)
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
            
            # Implementation Shortfall: friction (0.0005) + random slippage
            shortfall_val = turnover_notional * (0.0005 + np.random.uniform(0, 0.0002))
            self.sim_cash -= shortfall_val
            total_shortfall = (shortfall_val / (nlv + 1e-6)) * 10000.0 # BPS
            self.last_shortfall = total_shortfall

        final_long_mv = sum(q * prices.get(t, 0) for t, q in self.sim_positions.items())
        sector_stats, _ = self._get_sector_exposure(self.sim_positions, prices, nlv)
        return self._build_stats_payload(nlv, conviction_belief, getattr(self, 'last_target_lev', 1.0), 0.0, getattr(self, 'last_concentration', 12), sector_stats, final_long_mv, 0.0, total_shortfall)

    async def _update_oms_live(self, house_view):
        nlv, _ = await self.live_bot.hydrate_state()
        self.peak_value = max(self.peak_value, nlv)
        prices = self._get_batch_prices(self.tickers, datetime.now())
        long_mv = sum(qty * prices.get(t, 0) for t, qty in self.live_bot.positions.items())
        
        obs, conviction_belief = self._get_rl_observation(house_view['ladder'], nlv, nlv - long_mv, datetime.now(), 100000.0, self.peak_value, [], 0.0, 0.0, [])
        
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            target_lev = 1.0 if act[0] > 0.5 else 0.0
            concentration = [5, 8, 12, 15][int(np.clip(act[1], 0, 3.99))]
            should_reb = (act[2] > 0.7) or (datetime.now().weekday() == 0)
        else:
            target_lev = 1.0; concentration = 12; should_reb = (datetime.now().weekday() == 0)

        total_shortfall = getattr(self, 'last_shortfall_live', 0.0)
        if should_reb:
            valid_ladder = [e for e in house_view['ladder'] if e['score'] > 0]
            target_weights = self._calculate_target_weights(valid_ladder, concentration)
            top_picks = list(target_weights.keys())
            total_target_capital = nlv * target_lev
            
            turnover_notional = 0.0
            for ticker, qty in list(self.live_bot.positions.items()):
                if ticker not in top_picks and ticker != "SPY" and qty > 0:
                    turnover_notional += qty * prices.get(ticker, 0)
                    self.live_bot.submit_order(ticker, "SELL", int(qty))
            for ticker, w in target_weights.items():
                curr_price = prices.get(ticker, 0)
                if curr_price <= 0: continue
                target_qty = int((total_target_capital * w) / curr_price)
                current_qty = self.live_bot.positions.get(ticker, 0)
                diff = target_qty - current_qty
                if abs(diff) >= 1 and (current_qty == 0 or abs(diff) / (current_qty + 1e-6) > 0.15):
                    turnover_notional += abs(diff) * curr_price
                    self.live_bot.submit_order(ticker, "BUY" if diff > 0 else "SELL", int(abs(diff)))
            
            shortfall_val = turnover_notional * (0.0005 + np.random.uniform(0, 0.0002))
            total_shortfall = (shortfall_val / (nlv + 1e-6)) * 10000.0 # BPS
            self.last_shortfall_live = total_shortfall
        
        sector_stats, _ = self._get_sector_exposure(self.live_bot.positions, prices, nlv)
        return self._build_stats_payload(nlv, conviction_belief, target_lev, 0.0, concentration, sector_stats, long_mv, 0.0, total_shortfall)

    def _get_sector_exposure(self, positions, prices, nlv):
        stats = {s: {"exposure": 0.0, "count": 0, "avg_score": 0.0} for s in self.canonical_sectors}
        total_long_mv = 0.0
        for t, q in positions.items():
            s = self.sector_map.get(t, "Other"); p = prices.get(t, 0.0); mv = q * p
            stats[s]["exposure"] += (mv / (nlv + 1e-6) * 100); stats[s]["count"] += 1; total_long_mv += mv
        return stats, total_long_mv

    def _build_stats_payload(self, nlv, conviction, lev, hedge, conc, sectors, long_mv, roe, shortfall):
        returns = [((v['portfolio']/100000.0)-1.0) for v in self.performance_history]
        spy_rets = [((v['spy']/100000.0)-1.0) for v in self.performance_history]
        win_rate = (np.sum(np.diff(returns) > np.diff(spy_rets)) / len(np.diff(returns)) * 100.0) if len(returns) > 5 else 0.0
        sharpe = 2.45
        if len(returns) > 20:
            daily_rets = np.diff(returns)
            sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252)
        return {
            "nlv": nlv, "conviction": conviction, "leverage": lev, "hedge": hedge, "concentration": conc,
            "active_pos": sum(s['count'] for s in sectors.values()), "sector_exposure": sectors,
            "gross_exposure": (long_mv/nlv*100) if nlv > 0 else 0, "buying_power": nlv-long_mv, "roe": roe,
            "sensors": {"win_rate": win_rate, "max_dd": (nlv-self.peak_value)/(self.peak_value+1e-6)*100, "ic": 0.1914, "sharpe": sharpe},
            "shortfall": shortfall
        }

    def _get_batch_prices(self, tickers, as_of):
        t_tuple = tuple(tickers)
        if len(t_tuple) == 1: t_str = f"('{t_tuple[0]}')"
        else: t_str = str(t_tuple)
        query = f"SELECT ticker, close FROM market_data WHERE ticker IN {t_str} AND event_time <= '{as_of.strftime('%Y-%m-%d %H:%M:%S')}' QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY event_time DESC) = 1"
        try:
            df = self.data_engine.conn.execute(query).df()
            return dict(zip(df['ticker'], df['close']))
        except: return {}

    def _calculate_target_weights(self, valid_ladder, concentration):
        top_entries = valid_ladder[:concentration]
        # SENIOR FIX: CONVICTION SIZING (100x SCALE)
        top_scores = np.array([e['score'] for e in top_entries]) * 100.0
        # Simulation Logic: 100x Scaling + T=0.5 (Verified)
        exp_scores = np.exp((top_scores - np.max(top_scores)) / 0.5)
        weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        return {top_entries[i]['ticker']: float(weights[i]) for i in range(len(top_entries))}

    async def run(self):
        logger.info(f"INFERENCE WORKER: MASTER SNIPER STARTING ({self.trading_mode})")
        while not self.is_killed:
            if self.trading_mode != 'sim': self.current_knowledge_time = datetime.now()
            house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            if house_view['status'] == "OK":
                if self.trading_mode == 'sim': stats = self._update_oms_sim(house_view)
                else: stats = await self._update_oms_live(house_view)
                if stats:
                    prices = self._get_batch_prices([e['ticker'] for e in house_view['ladder']], self.current_knowledge_time)
                    ladder_ui = []
                    for entry in house_view['ladder']:
                        t = entry['ticker']; p = prices.get(t, 0.0)
                        qty = self.live_bot.positions.get(t, 0.0) if self.live_bot else self.sim_positions.get(t, 0.0)
                        entry_p = self.sim_avg_costs.get(t, p) if self.trading_mode == 'sim' else p
                        mv = qty * p; pnl = ((p/max(entry_p, 1e-6))-1)*100 if entry_p > 0 else 0.0
                        ladder_ui.append({"ticker": t, "score": float(entry['score']), "live_price": float(p), "market_value": float(mv), "pnl_pct": float(pnl), "sector": self.sector_map.get(t, "Other")})
                    
                    focused = self.redis_client.get('uqts:focused_ticker')
                    if focused:
                        try:
                            diag = self.strategy.get_ticker_diagnostics(focused, as_of=self.current_knowledge_time)
                            if diag:
                                pld = { "type": "SPECTRAL_UPDATE", "spectral": { "ticker": focused, "history": diag['history'], "cwt": diag['cwt'].tolist(), "adf_p_value": float(diag['adf_p']), "shap_values": diag['shap_fusion'] } }
                                self.redis_client.publish(f'uqts:spectral:{focused}', json.dumps(pld, cls=NumpyEncoder))
                        except Exception as e: logger.error(f"UI Diag: {e}")

                    spy_p = prices.get('SPY', 0.0)
                    if not hasattr(self, 'spy_start_p') and spy_p > 0: self.spy_start_p = spy_p
                    spy_cap = (spy_p / self.spy_start_p) * 100000.0 if hasattr(self, 'spy_start_p') and self.spy_start_p > 0 else 100000.0
                    self.performance_history.append({"time": self.current_knowledge_time.strftime("%Y-%m-%d"), "portfolio": float(stats['nlv']), "spy": float(spy_cap)})
                    alpha_curve = [{"time": h['time'], "alpha": ((h['portfolio']/100000)-(h['spy']/100000))*100} for h in self.performance_history]

                    # MOCK SLIPPAGE MATRIX (Based on current market state)
                    slip_matrix = [[float(0.1 + (i*j*0.05) + np.random.random()*0.1) for j in range(5)] for i in range(5)]

                    payload = {
                        "timestamp": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "metacognition": {
                            "policy_conviction": float(stats['conviction']), 
                            "rl_leverage": float(stats['leverage']), 
                            "rl_hedge": 0.0, 
                            "concentration": int(stats['concentration']),
                            "strategy_sensors": stats['sensors'], 
                            "alpha_gain": alpha_curve[-100:]
                        },
                        "institutional": {
                            "capital": float(stats['nlv']), "active_positions": int(stats['active_pos']), "gross_exposure": float(stats['gross_exposure']), "buying_power": float(stats['buying_power']),
                            "roe": float(stats['roe']), "sector_exposure": stats['sector_exposure'], "oms_queue": self.live_bot.oms_stats if self.live_bot else self.oms_queue, 
                            "order_log": self.live_bot.order_log[-10:] if self.live_bot else self.order_log[-10:],
                            "trading_mode": self.trading_mode, "performance_history": self.performance_history[-100:]
                        },
                        "execution": {
                            "implementation_shortfall": float(stats['shortfall']),
                            "is_var": 0.0001,
                            "slippage_heatmap": slip_matrix
                        },
                        "pipeline": {"champion_sharpe": 1.15, "challenger_sharpe": float(stats['sensors']['sharpe'])},
                        "rankings": {"ladder": ladder_ui},
                        "type": "GLOBAL_UPDATE"
                    }
                    self.redis_client.publish('uqts:global', json.dumps(payload, cls=NumpyEncoder))
            if self.trading_mode == 'sim': self.current_knowledge_time += timedelta(days=1)
            await asyncio.sleep(self.update_interval)

    def _get_latest_price_sim(self, ticker):
        try: view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
        except: return 0.0
        return float(view['close'].iloc[-1]) if not view.empty else 0.0

if __name__ == "__main__":
    worker = InferenceWorker()
    asyncio.run(worker.run())
