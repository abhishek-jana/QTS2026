import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from stable_baselines3 import PPO
from scipy.stats import spearmanr

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.data_engine import DataEngine
from research_lab.alpha_ranker_sniper import SniperRanker

class SimulationEngineV5:
    """
    Expert-Grade Simulation Engine (Ferrari Edition).
    V7.4.3: Professional Audit Synchronization
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['universe']['tickers']
        self.db_path = self.config['data_engine']['storage_path']
        self.engine = DataEngine(storage_path=self.db_path)
        self.universe = AlphaUniverse(conn=self.engine.conn, config=self.config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self.config['model_pipeline'].get('model_path', 'models/challenger_v2.pt')
        
        n_scales = len(self.config['signal_physics']['wavelet_transform']['scales'])
        specs = {
            'static': {'x_static': 1},
            'past': {
                'x_seq': 1,
                'x_spatial': n_scales,
                'x_volume': 1,
                'x_momentum': 3,
                'x_calendar': 4
            }
        }
        hidden_dim = self.config.get('model_pipeline', {}).get('architecture', {}).get('hidden_dim', 128)
        self.model = SniperRanker(specs=specs, hidden_dim=hidden_dim).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            logger.warning(f"Failed to load model weights from {model_path}: {e}")
            
        self.model.eval()

        self.rl_pilot = None
        rl_path = "models/rl_pilot_final.zip"
        if os.path.exists(rl_path):
            self.rl_pilot = PPO.load(rl_path, device="cpu")
            logger.info("SimulationEngine: RL Pilot Loaded (CPU).")
            
        self.vols_df = None
        self.latency_stress_test = self.config.get('execution_muscle', {}).get('latency_stress_test', False)

    def _get_batch_scores(self, steps):
        logger.info(f"🚀 SimulationEngine: Pre-computing Batch Inference on {len(steps)} steps...")
        scores_map = {}
        for step in tqdm(steps, desc="🧠 AI Thinking"):
            batch = step['batch'].to(self.device)
            with torch.no_grad():
                out_tensor = self.model(batch)
                if out_tensor.dim() > 1 and out_tensor.shape[1] > 1:
                    s = out_tensor[:, 1].cpu().numpy()
                else:
                    s = out_tensor.squeeze().cpu().numpy()
                scores_map[step['date']] = {t: float(val) for t, val in zip(batch.tickers, s)}
        return scores_map

    def _get_rl_observation(self, scores_list, nlv, cash, spy_df, current_dt, starting_capital, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history):
        # UNIFIED PERCEPTION: Scale scores by 100x to match training
        scores_np = np.array(scores_list) * 100.0
        sorted_scores = np.sort(scores_np)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        safe_nlv = max(nlv, 1.0)
        drawdown = (nlv - peak_value) / (peak_value + 1e-6)
        current_dt_naive = current_dt.replace(tzinfo=None)
        spy_mask_yesterday = spy_df['event_time'].dt.tz_localize(None) <= current_dt_naive
        if not spy_mask_yesterday.any(): return np.zeros(32, dtype=np.float32)
        spy_row = spy_df[spy_mask_yesterday].iloc[-1]
        
        belief = np.mean(top_10)
        vol = spy_row.get('vol_21', 0.02)
        
        spy_slice = spy_df[spy_mask_yesterday]
        if len(spy_slice) > 5:
            vol_vel = (spy_row['vol_21'] - spy_slice.iloc[-5]['vol_21']) * 1000.0
        else:
            vol_vel = 0.0
            
        long_mv = nlv - cash
        current_lev = (abs(long_mv)) / safe_nlv
        spy_trend = (spy_row.get('ma_ratio', 1.0) - 1.0) * 10.0
        rsi = (spy_row.get('rsi_14', 50.0) - 50.0) / 50.0
        spy_ret_yest = spy_row.get('ret', 0.0) * 10.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, vol, current_lev],
            [vol_vel, spy_trend, rsi, spy_ret_yest],
            [cash/safe_nlv, 0.0, 1.0, current_dt_naive.weekday()/6.0]
        ]).astype(np.float32)
        
        return np.clip(np.nan_to_num(obs), -10.0, 10.0)

    def run(self, start_date, end_date, max_leverage=1.0, backtest_mode=False):
        logger.info(f"🏁 Starting REAL-WORLD Simulation: {start_date.date()} -> {end_date.date()} | Max Lev: {max_leverage}x")
        self.engine.close()
        steps = self.universe.walk_forward(self.tickers, start_date, end_date, stride=1, latest_only=True, backtest_mode=backtest_mode)
        if not steps: return None
        all_scores = self._get_batch_scores(steps)
        conn = self.universe.conn
        spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY'").df()
        spy_df['event_time'] = pd.to_datetime(spy_df['event_time']); spy_df = spy_df.sort_values('event_time')
        spy_df['ret'] = spy_df['close'].pct_change(); spy_df['vol_21'] = spy_df['ret'].rolling(21).std(); spy_df['vov_21'] = spy_df['vol_21'].rolling(21).std()
        spy_df['ma_50'] = spy_df['close'].rolling(50).mean(); spy_df['ma_200'] = spy_df['close'].rolling(200).mean(); spy_df['ma_ratio'] = spy_df['ma_50'] / spy_df['ma_200']
        delta = spy_df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        spy_df['rsi_14'] = 100 - (100 / (1 + (gain/(loss+1e-6)))); spy_df = spy_df.ffill().fillna(0)
        spy_start_p = spy_df[spy_df['event_time'].dt.tz_localize(None) >= start_date.replace(tzinfo=None)].iloc[0]['close']
        self.last_steps = steps; self.last_spy_df = spy_df
        
        cash = 100000.0; positions = {}; price_cache = {}; starting_capital = 100000.0; peak_value = 100000.0
        portfolio_returns = []; score_history = []; history = []
        ic_buffer = []; realized_ic = 0.1914; wins = 0; total_fees = 0.0
        signal_queue = None; last_target_lev = None; last_concentration = 12

        for i in range(len(steps)):
            dt = steps[i]['date']; batch = steps[i]['batch']; spy_p = spy_df[spy_df['event_time'].dt.tz_localize(None) <= dt.replace(tzinfo=None)]['close'].iloc[-1]
            # Update price memory
            for t in batch.tickers: price_cache[t] = float(batch.data['raw_price'][batch.tickers.index(t)])
            
            pos_mv = sum([qty * price_cache.get(t, 0.0) for t, qty in positions.items()])
            nlv = cash + pos_mv; peak_value = max(peak_value, nlv)
            spy_nlv = (spy_p / spy_start_p) * 100000
            
            if i > 0: 
                prev_nlv = history[-1]['NLV']
                agent_ret = (nlv / prev_nlv) - 1
                spy_ret = (spy_nlv / history[-1]['SPY_NLV']) - 1
                if agent_ret > spy_ret: wins += 1
                portfolio_returns.append(agent_ret)

            scores_dict = all_scores.get(dt, {}); obs_scores = [scores_dict.get(t, 0.0) for t in self.tickers]; score_history.append(obs_scores)
            obs = self._get_rl_observation(obs_scores, nlv, cash, spy_df, dt, starting_capital, peak_value, portfolio_returns, hedge_qty=0, hedge_entry_p=0, score_history=score_history)
            
            # Realized IC
            ic_buffer.append({"scores": scores_dict, "prices": price_cache.copy()})
            if len(ic_buffer) > 3:
                past = ic_buffer.pop(0)
                r_rets = []; p_scores = []
                for t in self.tickers:
                    if t in price_cache and t in past['prices']:
                        r_rets.append((price_cache[t]/(past['prices'][t]+1e-9))-1)
                        p_scores.append(past['scores'].get(t, 0.0))
                if len(r_rets) > 10:
                    val_ic, _ = spearmanr(p_scores, r_rets)
                    realized_ic = max(0, val_ic)

            # RL Decision for TODAY
            if self.rl_pilot:
                action, _ = self.rl_pilot.predict(obs, deterministic=True)
                should_reb_signal = (action[2] > 0.7) or (dt.weekday() == 0) or (i == 0)
                target_lev_signal = 1.0 if action[0] > 0.5 else 0.0
                concentration_signal = [5, 8, 12, 15][int(np.clip(action[1], 0, 3.99))]
            else:
                should_reb_signal = (dt.weekday() == 0) or (i == 0); target_lev_signal = 1.0; concentration_signal = 12

            current_decision = {"should_reb": should_reb_signal, "target_lev": target_lev_signal, "concentration": concentration_signal, "scores": scores_dict}

            # Latency Stress Test (Sync with InferenceWorker)
            if self.latency_stress_test:
                decision_to_execute = signal_queue
                signal_queue = current_decision
                if decision_to_execute is None:
                    wr_step = (wins / (i+1)) * 100.0 if i > 0 else 0.0
                    history.append({"Date": dt, "NLV": nlv, "SPY_NLV": spy_nlv, "IC": realized_ic, "WinRate": wr_step, "Lev": 0.0, "Conc": 5})
                    continue
            else:
                decision_to_execute = current_decision

            # Execute decision
            should_rebalance = decision_to_execute['should_reb']
            target_lev = decision_to_execute['target_lev']
            concentration = decision_to_execute['concentration']
            exec_scores = decision_to_execute['scores']

            if should_rebalance:
                last_target_lev = target_lev; last_concentration = concentration
                target_notional = (nlv * target_lev); top_picks = sorted(exec_scores.keys(), key=lambda x: exec_scores.get(x, -9), reverse=True)[:concentration]
                top_scores = np.array([exec_scores.get(x, -9) for x in top_picks]) * 100.0
                exp_scores = np.exp((top_scores - np.max(top_scores)) / 0.5)
                weights = exp_scores / (np.sum(exp_scores) + 1e-9)
                
                turnover_notional = 0.0
                for t in list(positions.keys()):
                    if t not in top_picks: 
                        p = price_cache.get(t, 0.0); v = positions[t] * p
                        cash += v; turnover_notional += v; del positions[t]

                for idx_w, t in enumerate(top_picks):
                    p = price_cache.get(t, 0.0)
                    if p > 0:
                        t_qty = int((target_notional * weights[idx_w]) / p); c_qty = positions.get(t, 0)
                        if c_qty == 0 or abs(t_qty - c_qty) / (c_qty + 1e-6) > 0.15:
                            diff_qty = t_qty - c_qty; t_v = diff_qty * p
                            cash -= t_v; turnover_notional += abs(t_v); positions[t] = t_qty
                
                fee_step = turnover_notional * 0.0005; cash -= fee_step; total_fees += fee_step

            wr = (wins / (i+1)) * 100.0 if i > 0 else 0.0
            if i % 20 == 0 or i == len(steps) - 1:
                logger.debug(f"[{dt.date()}] NLV: ${nlv:,.2f} | SPY: ${spy_nlv:,.2f} | WinRate: {wr:.1f}% | IC: {realized_ic:.3f} | Pos: {len(positions)}")
            history.append({"Date": dt, "NLV": nlv, "SPY_NLV": spy_nlv, "IC": realized_ic, "WinRate": wr, "Lev": last_target_lev or 0.0, "Conc": last_concentration})

        df = pd.DataFrame(history)
        logger.success(f"Simulation finished. Final Capital: ${nlv:,.2f} | Total Fees: ${total_fees:,.2f}"); return df
