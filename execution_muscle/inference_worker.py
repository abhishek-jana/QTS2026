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
        self.canonical_sectors = ["Tech", "Semi", "Healthcare", "Financial", "Retail", "Auto", "Energy", "Materials", "Industrials", "Index", "Other"]
        
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'], read_only=True)
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        self.spy_df = self.data_engine.conn.execute("SELECT event_time, close FROM market_data WHERE ticker = 'SPY'").df()
        self.spy_df['event_time'] = pd.to_datetime(self.spy_df['event_time'])
        self.spy_df = self.spy_df.sort_values('event_time')
        self.spy_df['ret'] = self.spy_df['close'].pct_change()
        self.spy_df['vol_21'] = self.spy_df['ret'].rolling(21).std()
        self.spy_df['ma_ratio'] = self.spy_df['close'].rolling(50).mean() / (self.spy_df['close'].rolling(200).mean() + 1e-9)
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
        from collections import deque
        self.ic_buffer = deque(maxlen=4)
        self.alpha_history = []
        self.realized_ic = 0.0
        self.is_killed = False

    def initialize(self):
        start_date = datetime(2024, 1, 1)
        mask = self.spy_df['event_time'].dt.tz_localize(None) >= start_date.replace(tzinfo=None)
        
        if self.trading_mode != 'sim':
            saved_spy_p = self.redis_client.get('uqts:live:spy_start_p')
            if saved_spy_p: self.spy_start_p = float(saved_spy_p)
            else:
                curr_prices = self._get_batch_prices(['SPY'], datetime.now())
                self.spy_start_p = float(curr_prices.get('SPY', self.spy_df.iloc[-1]['close']))
                self.redis_client.set('uqts:live:spy_start_p', float(self.spy_start_p))
        else:
            self.spy_start_p = float(self.spy_df[mask].iloc[0]['close']) if not self.spy_df[mask].empty else 1.0

        if self.trading_mode == 'sim':
            end_date = datetime.now()
            steps = self.strategy.lab.walk_forward(self.tickers, start_date, end_date, stride=1)
            self.sim_rankings_cache = {}
            from research_lab.alpha_universe import MultiModalBatch
            device = next(self.strategy.model.parameters()).device
            with torch.no_grad():
                for s in tqdm(steps, desc="Pre-computing AI Rankings"):
                    dt_key = s['date'].strftime("%Y-%m-%d")
                    batch = s['batch'].to(device)
                    out = self.strategy.model(batch)
                    scores = out[:, 1].detach().cpu().numpy()
                    ladder = [{"ticker": t, "score": float(scores[i]), "price": float(batch.data['raw_price'][i].item())} for i, t in enumerate(batch.tickers)]
                    ladder.sort(key=lambda x: x['score'], reverse=True)
                    self.sim_rankings_cache[dt_key] = {"ladder": ladder, "status": "OK"}
        
        self.live_bot = AsyncPaperBot(self.config, 100000.0) if self.trading_mode != 'sim' else None
        self.is_initialized = True

    def _get_rl_observation(self, ladder, nlv, cash, current_dt, starting_capital, peak_value):
        scores_np = np.array([e['score'] for e in ladder]) * 100.0
        sorted_scores = np.sort(scores_np)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        drawdown = (nlv - peak_value) / (peak_value + 1e-6)
        spy_mask = self.spy_df['event_time'].dt.tz_localize(None) <= current_dt.replace(tzinfo=None)
        spy_row = self.spy_df[spy_mask].iloc[-1]
        
        # Use Bayesian belief if strategy engine has it, else mean of top 10
        belief = self.strategy.meta_controller.get_position_scaler() * 100.0
        
        spy_slice = self.spy_df[spy_mask]
        vol_vel = (spy_row['vol_21'] - spy_slice.iloc[-5]['vol_21']) * 1000.0 if len(spy_slice) > 5 else 0.0
        
        return build_rl_observation(
            top_10_scores=top_10, bot_10_scores=bot_10, belief=belief, drawdown=drawdown,
            vol_21=spy_row['vol_21'], current_lev=abs(nlv-cash)/nlv, vol_vel=vol_vel,
            spy_trend=(spy_row['ma_ratio']-1.0)*10.0, rsi=(spy_row['rsi_14']-50.0)/50.0,
            spy_ret=spy_row['ret'], cash_ratio=cash/nlv, dow=current_dt.weekday()/6.0
        ), belief/100.0

    def _update_oms_sim(self, house_view):
        pos_mv = sum(qty * self.sim_price_memory.get(t, 0) for t, qty in self.sim_positions.items())
        nlv = self.sim_cash + pos_mv
        self.peak_value = max(self.peak_value, nlv)
        
        obs, conviction = self._get_rl_observation(house_view['ladder'], nlv, self.sim_cash, self.current_knowledge_time, 100000.0, self.peak_value)
        
        if self.rl_pilot:
            act, _ = self.rl_pilot.predict(obs, deterministic=True)
            should_reb = (act[2] > 0.7) or (self.current_knowledge_time.weekday() == 0)
            target_lev = float(np.clip(act[0], 0.0, 1.0))
            concentration = [5, 10, 15, 20][int(np.clip(act[1], 0, 3.99))]
        else:
            should_reb = (self.current_knowledge_time.weekday() == 0); target_lev = 1.0; concentration = 12

        if should_reb:
            self.last_target_lev = target_lev; self.last_concentration = concentration
            asset_cap = self.config.get('execution_muscle', {}).get('max_single_asset_cap', 0.15)
            temp = self.config.get('execution_muscle', {}).get('allocation_temperature', 0.1)
            
            # Use shared weights calculator
            scores_dict = {e['ticker']: e['score'] for e in house_view['ladder']}
            scores_arr = np.array([scores_dict.get(t, 0) for t in self.tickers])
            
            top_k_idx, weights = calculate_safe_weights(scores_arr, concentration, asset_cap, temp)
            target_weights = {self.tickers[idx]: weights[i] for i, idx in enumerate(top_k_idx)}
            
            # Execute trades
            target_notional = nlv * target_lev
            for t in list(self.sim_positions.keys()):
                if t not in target_weights:
                    p = self.sim_price_memory.get(t, 0); v = self.sim_positions[t] * p
                    self.sim_cash += v; del self.sim_positions[t]
            
            for t, w in target_weights.items():
                p = self.sim_price_memory.get(t, 0)
                if p > 0:
                    t_qty = int((target_notional * w) / p)
                    diff = t_qty - self.sim_positions.get(t, 0)
                    self.sim_cash -= diff * p; self.sim_positions[t] = t_qty
            
            self.sim_cash -= (nlv * target_lev * 0.0005) # Simple fee model

        return {"nlv": nlv, "conviction": conviction, "leverage": target_lev, "concentration": concentration, "active_pos": len(self.sim_positions), "sector_exposure": {}, "gross_exposure": abs(nlv-self.sim_cash)/nlv, "buying_power": self.sim_cash, "roe": (nlv/100000.0-1)*100, "shortfall": 0.0, "sensors": {"win_rate": 0, "max_dd": (nlv-self.peak_value)/self.peak_value*100, "ic": 0, "sharpe": 0}}

    def _get_batch_prices(self, tickers, as_of):
        t_str = "('" + "','".join(tickers) + "')"
        query = f"SELECT ticker, close FROM market_data WHERE ticker IN {t_str} AND event_time <= '{as_of}' QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY event_time DESC) = 1"
        df = self.data_engine.conn.execute(query).df()
        return dict(zip(df['ticker'], df['close']))

    async def run(self):
        if self.live_bot: asyncio.create_task(self.live_bot.run_stream())
        while not self.is_killed:
            if not self.is_initialized: await asyncio.sleep(1); continue
            
            dt_key = self.current_knowledge_time.strftime("%Y-%m-%d")
            house_view = self.sim_rankings_cache.get(dt_key, {"status": "DATA_MISSING", "ladder": []})
            
            if self.trading_mode == 'sim':
                for e in house_view['ladder']: self.sim_price_memory[e['ticker']] = e['price']
                stats = self._update_oms_sim(house_view)
                self.performance_history.append({"time": dt_key, "portfolio": stats['nlv'], "spy": (self._get_batch_prices(['SPY'], self.current_knowledge_time).get('SPY', 1.0)/self.spy_start_p)*100000.0})
                self.current_knowledge_time += timedelta(days=1)
                if self.current_knowledge_time > datetime.now(): self.is_killed = True
            
            await asyncio.sleep(self.update_interval)

if __name__ == "__main__":
    worker = InferenceWorker()
    asyncio.run(worker.run())
