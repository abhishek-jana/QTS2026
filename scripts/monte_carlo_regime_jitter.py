import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import duckdb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from stable_baselines3 import PPO

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.data_engine import DataEngine
from research_lab.alpha_ranker_sniper import SniperRanker

class MonteCarloStressTest:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['universe']['tickers']
        self.db_path = self.config['data_engine']['storage_path']
        self.engine = DataEngine(storage_path=self.db_path)
        self.universe = AlphaUniverse(conn=self.engine.conn, config=self.config)
        
        # Hardware Aware
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
        
        logger.info(f"MonteCarlo: Loading SniperRanker weights from {model_path}...")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            logger.warning(f"Failed to load model weights: {e}")
        self.model.eval()

        self.rl_pilot = None
        rl_path = "models/rl_pilot_final.zip"
        if os.path.exists(rl_path):
            self.rl_pilot = PPO.load(rl_path, device="cpu")
            logger.info("MonteCarlo: RL Pilot Loaded (CPU).")
            
        # Load Pre-computed Volatilities for Level 3
        vol_path = "data/rl/train_stock_vols.csv"
        if os.path.exists(vol_path):
            self.vols_df = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            self.vols_df = self.vols_df.sort_index().loc[~self.vols_df.index.duplicated(keep='first')].ffill().bfill().fillna(0.02)
        else:
            logger.warning("train_stock_vols.csv not found. Falling back to constant volatility.")
            self.vols_df = None

    def _get_rl_observation(self, scores_list, nlv, cash, spy_df, current_dt, starting_capital, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history):
        # UNIFIED PERCEPTION: Scale scores by 100x
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
        
        # VOL VELOCITY - Scale by 1000x
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

    def run_one_path(self, steps, spy_df, price_pivot, jitter_scale=0.001, jump_prob=0.01, max_leverage=1.0):
        """
        Real-World Monte Carlo (RWMC) Path:
        - Jump-Diffusion: Poisson outliers.
        - Volatility-Scaled Friction.
        - Smart Execution Trigger.
        """
        cash = 100000.0; positions = {}; peak_value = 100000.0; hedge_qty = 0.0; hedge_entry_p = 0.0
        portfolio_returns = []; score_history = []; history = []
        last_lev = 1.0; last_conc = 12

        for i in range(len(steps)):
            dt = steps[i]['date']; batch = steps[i]['batch']; current_dt_naive = dt.replace(tzinfo=None)
            jump_multiplier = 1.0; is_jump = np.random.random() < jump_prob
            if is_jump: jump_multiplier = 1.0 + np.random.uniform(-0.07, 0.03)
            
            prices = price_pivot.asof(current_dt_naive); pos_mv = sum(q * (prices.get(t, 0.0) * jump_multiplier) for t, q in positions.items())
            spy_slice = spy_df[spy_df['event_time'].dt.tz_localize(None) <= current_dt_naive]
            spy_p = (spy_slice.iloc[-1]['close'] if not spy_slice.empty else 450.0) * jump_multiplier
            nlv = cash + pos_mv; peak_value = max(peak_value, nlv)
            if i > 0: portfolio_returns.append((nlv - history[-1]) / history[-1])

            with torch.no_grad():
                out = self.model(batch.to(self.device))
                raw_s = out[:, 1].cpu().numpy() if out.shape[1] > 1 else out.squeeze().cpu().numpy()
                scores = raw_s + np.random.normal(0, jitter_scale, len(raw_s))
                scores_dict = {t: float(val) for t, val in zip(batch.tickers, scores)}

            obs_scores = [scores_dict.get(t, 0.0) for t in self.tickers]; score_history.append(obs_scores)
            
            if self.rl_pilot:
                obs = self._get_rl_observation(obs_scores, nlv, cash, spy_df, dt, 100000.0, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history)
                act, _ = self.rl_pilot.predict(obs, deterministic=True)
                # V7.4: [Risk Toggle, Concentration, Trigger]
                should_rebalance = (act[2] > 0.7) or (dt.weekday() == 0)
                if should_rebalance:
                    last_lev = 1.0 if act[0] > 0.5 else 0.0
                    last_conc = [5, 8, 12, 15][int(np.clip(act[1], 0, 3.99))]
            else:
                should_rebalance = (dt.weekday() == 0); last_lev = 1.0; last_conc = 12

            if should_rebalance:
                target_notional = (nlv * last_lev); top_picks = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)[:last_conc]
                top_scores = np.array([scores_dict.get(x, -9) for x in top_picks]); exp_scores = np.exp((top_scores - np.max(top_scores)) / 0.5)
                weights = np.clip(exp_scores / (np.sum(exp_scores) + 1e-9), 0.0, 1.0); weights = weights / (np.sum(weights) + 1e-9)
                fric = 0.0050 if is_jump and jump_multiplier < 1.0 else 0.0005
                turnover = 0.0; trade_prices = prices * jump_multiplier
                for t in list(positions.keys()):
                    if t not in top_picks: v = positions[t] * trade_prices.get(t, 0.0); cash += v; turnover += v; del positions[t]
                for idx, t in enumerate(top_picks):
                    p = trade_prices.get(t, 0.0)
                    if p > 0:
                        t_qty = int((target_notional * weights[idx]) / p); c_qty = positions.get(t, 0)
                        if c_qty == 0 or abs(t_qty - c_qty) / (c_qty + 1e-6) > 0.15:
                            t_v = (t_qty - c_qty) * p; cash -= t_v; turnover += abs(t_v); positions[t] = t_qty
                cash -= turnover * fric

            history.append(nlv)
        return history

    def run_simulation(self, n_paths=20, backtest_mode=False, steps=None, spy_df=None):
        logger.info(f"🎲 Initiating Monte Carlo Stress Test: {n_paths} Synthetic Regimes...")
        dates = [s['date'] for s in steps]
        conn = self.universe.conn
        if spy_df is None:
            spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY'").df()
            spy_df['event_time'] = pd.to_datetime(spy_df['event_time']); spy_df = spy_df.sort_values('event_time')
            spy_df['ret'] = spy_df['close'].pct_change(); spy_df['vol_21'] = spy_df['ret'].rolling(21).std(); spy_df['vov_21'] = spy_df['vol_21'].rolling(21).std()
            spy_df['ma_50'] = spy_df['close'].rolling(50).mean(); spy_df['ma_200'] = spy_df['close'].rolling(200).mean(); spy_df['ma_ratio'] = spy_df['ma_50'] / spy_df['ma_200']
            delta = spy_df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            spy_df['rsi_14'] = 100 - (100 / (1 + (gain/(loss+1e-6)))); spy_df = spy_df.ffill().fillna(0)
        
        all_prices = conn.execute(f"SELECT ticker, event_time, close FROM market_data WHERE ticker IN {tuple(self.tickers)}").df()
        all_prices['event_time'] = pd.to_datetime(all_prices['event_time'])
        price_pivot = all_prices.drop_duplicates(subset=['event_time', 'ticker']).pivot(index='event_time', columns='ticker', values='close').ffill().bfill()
        
        all_paths = []
        for i in range(n_paths):
            logger.info(f"Simulation Path {i+1}/{n_paths}..."); all_paths.append(self.run_one_path(steps, spy_df, price_pivot, max_leverage=1.0))
            
        all_paths = np.array(all_paths); plt.figure(figsize=(12, 7), facecolor='#050505'); ax = plt.gca(); ax.set_facecolor('#050505')
        for path in all_paths: plt.plot(dates, path, color='#10b981', alpha=0.15, lw=1)
        mean_path = np.mean(all_paths, axis=0); plt.plot(dates, mean_path, color='#10b981', lw=3, label='RL Monte Carlo Mean')
        
        original_pilot = self.rl_pilot; self.rl_pilot = None
        baseline = self.run_one_path(steps, spy_df, price_pivot, jitter_scale=0.0, jump_prob=0.0, max_leverage=1.0)
        self.rl_pilot = original_pilot; plt.plot(dates, baseline, color='#3b82f6', lw=2, label='RankNet Baseline (Top 12)')
        
        spy_slice = spy_df[(spy_df['event_time'].dt.tz_localize(None) >= dates[0].replace(tzinfo=None)) & (spy_df['event_time'].dt.tz_localize(None) <= dates[-1].replace(tzinfo=None))]
        spy_line = (spy_slice['close'] / spy_slice.iloc[0]['close']) * 100000
        plt.plot(spy_slice['event_time'], spy_line, color='#475569', ls='--', label='SPY Benchmark')
        
        plt.title("V7.4 'Smart Sniper' Monte Carlo: Risk-Aware Performance", color='white', fontweight='black'); plt.legend(); plt.grid(True, alpha=0.1); plt.savefig("data/monte_carlo_robustness.png"); plt.close()
        logger.success("✅ Monte Carlo Stress Test Complete. Plot saved."); logger.info(f"Mean Final NLV: ${mean_path[-1]:,.2f} | Baseline NLV: ${baseline[-1]:,.2f} | SPY NLV: ${spy_line.iloc[-1]:,.2f}")

if __name__ == "__main__":
    np.random.seed(42); import random; random.seed(42); torch.manual_seed(42)
    test = MonteCarloStressTest(); test.run_simulation(n_paths=20)
