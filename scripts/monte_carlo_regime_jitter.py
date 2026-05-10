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
        
        logger.info(f"MonteCarlo: Loading RankNet to {self.device}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
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
        """Prepares the 32-sensor vector - EXACT MATCH to PortfolioGym V5.2."""
        sorted_scores = np.sort(scores_list)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        drawdown = (nlv - peak_value) / peak_value if peak_value > 0 else 0
        current_dt_naive = current_dt.replace(tzinfo=None)
        
        # V5.2 GHOST-PROTOCOL: Use T-1 (Yesterday) for ALL macro
        spy_mask_yesterday = spy_df['event_time'].dt.tz_localize(None) < current_dt_naive
        if not spy_mask_yesterday.any(): return np.zeros(32, dtype=np.float32)
        spy_prev = spy_df[spy_mask_yesterday].iloc[-1]
        
        belief = np.mean(top_10) - np.mean(bot_10)
        vol = spy_prev.get('vol_21', 0.0)
        long_mv = nlv - cash

        spy_prev_close = spy_df[spy_mask_yesterday].iloc[-1]['close'] if spy_mask_yesterday.any() else spy_df.iloc[0]['close']
        short_mv = (hedge_qty * spy_prev_close) if hedge_qty > 0 else 0
        current_lev = (abs(long_mv) + abs(short_mv)) / nlv if nlv > 0 else 0
        
        vov = spy_prev.get('vov_21', 0.0) * 100.0
        ma_ratio = (spy_prev.get('ma_ratio', 1.0) - 1.0) * 10.0
        rsi = (spy_prev.get('rsi_14', 50.0) - 50.0) / 50.0
        spy_ret_yest = spy_prev.get('ret', 0.0) * 10.0
        
        cash_ratio = cash / nlv if nlv > 0 else 1.0
        
        # Alpha Decay Calculation
        if len(score_history) > 5:
            prev_top_mean = np.mean(np.sort(score_history[-5])[-10:])
            alpha_decay = (np.mean(top_10) - prev_top_mean)
        else:
            alpha_decay = 0.0
            
        recent_p_rets = portfolio_returns[-5:]
        rs = np.mean(recent_p_rets) / (spy_prev.get('ret', 0.001) + 1e-6) if recent_p_rets else 1.0
        dow = current_dt_naive.weekday() / 6.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, vol, current_lev],
            [vov, ma_ratio, rsi, spy_ret_yest],
            [cash_ratio, alpha_decay, rs, dow]
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def run_one_path(self, steps, spy_df, jitter_scale=0.005, crash_prob=0.001):
        cash = 100000.0
        positions = {} 
        peak_value = 100000.0
        hedge_qty = 0.0
        hedge_entry_p = 0.0
        portfolio_returns = []
        score_history = []
        history = []
        
        for i in range(len(steps)):
            curr_step = steps[i]; dt = curr_step['date']; batch = curr_step['batch']
            crash_multiplier = 0.95 if np.random.random() < crash_prob else 1.0
            
            prices = {t: float(batch.data['raw_price'][batch.tickers.index(t)]) * crash_multiplier for t in batch.tickers}
            pos_mv = sum(q * prices.get(t, 0) for t, q in positions.items())
            spy_p = prices.get('SPY', spy_df[spy_df['event_time'].dt.tz_localize(None) <= dt.replace(tzinfo=None)]['close'].iloc[-1])
            hedge_pnl = hedge_qty * (hedge_entry_p - spy_p) if hedge_qty > 0 else 0.0
            
            nlv = cash + pos_mv + hedge_pnl
            peak_value = max(peak_value, nlv)
            if i > 0: portfolio_returns.append((nlv - history[-1]) / history[-1])

            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.data.items() if k in ['x_seq', 'x_spatial', 'x_graph', 'x_volume', 'x_momentum']}
                raw_scores = self.model(inputs).squeeze().cpu().numpy()
                if raw_scores.ndim == 0: raw_scores = np.array([raw_scores.item()])
                scores = raw_scores + np.random.normal(0, jitter_scale, len(raw_scores))
                scores_dict = {t: float(val) for t, val in zip(batch.tickers, scores)}

            obs_scores = [scores_dict.get(t, 0.0) for t in self.tickers]
            score_history.append(obs_scores)

            if self.rl_pilot:
                obs = self._get_rl_observation(obs_scores, nlv, cash, spy_df, dt, 100000.0, peak_value, portfolio_returns, hedge_qty, hedge_entry_p, score_history)
                action, _ = self.rl_pilot.predict(obs, deterministic=True)
                _, hedge_ratio, concentration_idx = action
                target_lev = 1.0
                total_gross = float(target_lev) + float(hedge_ratio)
                if total_gross > 1.0:
                    scale = 1.0 / total_gross
                    target_lev = float(target_lev) * scale
                    hedge_ratio = float(hedge_ratio) * scale
                concentration = [5, 8, 12][int(np.clip(concentration_idx, 0, 2.999))]
            else:
                target_lev = 1.0; hedge_ratio = 0.0; concentration = 5 
            
            if dt.weekday() == 0:
                friction = np.random.uniform(0.0010, 0.0025)
                target_notional = nlv * target_lev
                top_picks = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)[:concentration]
                
                # LEVEL 2/3: Sizing Logic
                top_scores = np.array([scores_dict.get(x, -9) for x in top_picks])
                temperature = 2.0
                exp_scores = np.exp((top_scores - np.max(top_scores)) / temperature)
                conviction_weights = exp_scores / np.sum(exp_scores)
                
                risk_parity = self.config.get('execution_muscle', {}).get('risk_parity_sizing', True)
                if risk_parity and self.vols_df is not None:
                    dt_naive = dt.replace(tzinfo=None)
                    vol_mask = self.vols_df.index <= dt_naive
                    if vol_mask.any():
                        vol_row = self.vols_df[vol_mask].iloc[-1]
                        stock_vols = np.array([vol_row.get(t, 0.02) for t in top_picks])
                    else:
                        stock_vols = np.array([0.02] * len(top_picks))
                    
                    risk_adjusted_weights = conviction_weights / (stock_vols + 1e-6)
                    weights = risk_adjusted_weights / np.sum(risk_adjusted_weights)
                else:
                    weights = conviction_weights
                    
                # SENIOR FIX: Increase cap to 50% to allow AI to ride winners
                weights = np.clip(weights, 0.0, 0.50)
                weights = weights / np.sum(weights)

                turnover_notional = 0.0

                for t in list(positions.keys()):
                    if t not in top_picks:
                        v = positions[t] * prices.get(t, 0); cash += v; turnover_notional += v; del positions[t]

                for idx_w, t in enumerate(top_picks):
                    p = prices.get(t, 0)
                    if p > 0:
                        target_qty = int((target_notional * weights[idx_w]) / p)
                        diff_qty = target_qty - positions.get(t, 0)
                        trade_v = diff_qty * p; cash -= trade_v; turnover_notional += abs(trade_v); positions[t] = target_qty

                cash += hedge_pnl
                new_hedge_qty = (nlv * hedge_ratio) / spy_p if spy_p > 0 else 0
                turnover_notional += abs((new_hedge_qty - hedge_qty) * spy_p)
                hedge_qty = new_hedge_qty; hedge_entry_p = spy_p
                # SENIOR FIX: Institutional friction 5bps (0.0005)
                cash -= turnover_notional * 0.0005 

            history.append(nlv)
        return history

    def run_simulation(self, n_paths=20, backtest_mode=False, steps=None, spy_df=None):
        logger.info(f"🎲 Initiating Monte Carlo Stress Test: {n_paths} Synthetic Regimes...")
        start_date = datetime(2024, 1, 1); end_date = datetime.now() - timedelta(days=1)
        
        if steps is None:
            steps = self.universe.walk_forward(self.tickers, start_date, end_date, stride=1, latest_only=True, backtest_mode=backtest_mode)
        if not steps: return
        
        if spy_df is None:
            conn = duckdb.connect(self.db_path, read_only=True)
            spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY'").df()
            spy_df['event_time'] = pd.to_datetime(spy_df['event_time']); spy_df = spy_df.sort_values('event_time')
            spy_df['ret'] = spy_df['close'].pct_change()
            spy_df['vol_21'] = spy_df['ret'].rolling(21).std()
            spy_df['vov_21'] = spy_df['vol_21'].rolling(21).std()
            spy_df['ma_50'] = spy_df['close'].rolling(50).mean()
            spy_df['ma_200'] = spy_df['close'].rolling(200).mean()
            spy_df['ma_ratio'] = spy_df['ma_50'] / spy_df['ma_200']
            delta = spy_df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            spy_df['rsi_14'] = 100 - (100 / (1 + (gain/loss))); spy_df = spy_df.ffill().fillna(0); conn.close()
        
        all_paths = []
        for i in range(n_paths):
            logger.info(f"Simulation Path {i+1}/{n_paths}..."); all_paths.append(self.run_one_path(steps, spy_df))
            
        all_paths = np.array(all_paths)
        plt.figure(figsize=(12, 7), facecolor='#050505'); ax = plt.gca(); ax.set_facecolor('#050505')
        dates = [s['date'] for s in steps]
        for path in all_paths: plt.plot(dates, path, color='#10b981', alpha=0.15, lw=1)
        mean_path = np.mean(all_paths, axis=0); plt.plot(dates, mean_path, color='#10b981', lw=3, label='Monte Carlo Mean')
        spy_start = spy_df[spy_df['event_time'].dt.tz_localize(None) >= start_date.replace(tzinfo=None)].iloc[0]['close']
        spy_slice = spy_df[(spy_df['event_time'].dt.tz_localize(None) >= start_date.replace(tzinfo=None)) & (spy_df['event_time'].dt.tz_localize(None) <= end_date.replace(tzinfo=None))]
        spy_line = (spy_slice['close'] / spy_start) * 100000
        plt.plot(spy_slice['event_time'], spy_line, color='#475569', ls='--', label='SPY Benchmark')
        plt.title("UQTS-2026 V5.0 Monte Carlo Stress Test", color='white', fontweight='black'); plt.legend(); plt.grid(True, alpha=0.1); plt.savefig("data/monte_carlo_robustness.png")
        plt.close()
        logger.success("✅ Monte Carlo Stress Test Complete. Plot saved to data/monte_carlo_robustness.png")
        logger.info(f"Mean Final NLV: ${mean_path[-1]:,.2f} | Worst Case (Min): ${np.min(all_paths[:,-1]):,.2f}")

if __name__ == "__main__":
    # SENIOR FIX: Default seed for standalone execution
    np.random.seed(42)
    import random
    random.seed(42)
    torch.manual_seed(42)
    
    test = MonteCarloStressTest()
    test.run_simulation(n_paths=20)
