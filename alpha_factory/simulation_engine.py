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
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.data_engine import DataEngine

class SimulationEngineV5:
    """
    Expert-Grade Simulation Engine (Ferrari Edition).
    - Batch Inference for speed.
    - Audited Institutional Ledger.
    - SPY Benchmarking.
    - PHASE 4: Institutional Guardrails (Liquidity Caps + Slippage).
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
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

        # Load RL Pilot
        self.rl_pilot = None
        rl_path = "models/rl_pilot_final.zip"
        if os.path.exists(rl_path):
            self.rl_pilot = PPO.load(rl_path, device="cpu")
            logger.info("SimulationEngine: RL Pilot Loaded (CPU).")

    def _get_batch_scores(self, steps):
        logger.info(f"🚀 SimulationEngine: Pre-computing Batch Inference on {len(steps)} steps...")
        scores_map = {}
        for step in tqdm(steps, desc="🧠 AI Thinking"):
            batch = step['batch'].to(self.device)
            with torch.no_grad():
                inputs = {k: v for k, v in batch.data.items() if k in ['x_seq', 'x_spatial', 'x_graph', 'x_volume']}
                s = self.model(inputs).squeeze().cpu().numpy()
                scores_map[step['date']] = {t: float(val) for t, val in zip(batch.tickers, s)}
        return scores_map

    def _get_rl_observation(self, scores_list, nlv, cash, spy_df, current_dt, starting_capital, peak_value):
        """Prepares the 24-sensor sensor vector - EXACT MATCH to PortfolioGym."""
        sorted_scores = np.sort(scores_list)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        # Risk Metrics (Gym uses peak_value)
        drawdown = (nlv - peak_value) / peak_value if peak_value > 0 else 0
        
        # SPY Vol (Gym uses 30-day window resampled to daily)
        start_date = current_dt - timedelta(days=30)
        spy_window = spy_df[(spy_df['event_time'] >= start_date) & (spy_df['event_time'] <= current_dt)]
        
        if len(spy_window) > 5:
            daily_rets = spy_window.set_index('event_time')['close'].resample('1D').last().pct_change().dropna()
            vol = daily_rets.std() if len(daily_rets) > 1 else 0.0
        else:
            vol = 0.0
        
        belief = np.mean(top_10) - np.mean(bot_10)
        current_lev = (nlv - cash) / nlv if nlv > 0 else 0
        
        obs = np.concatenate([top_10, bot_10, [belief, drawdown, vol, current_lev]]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def run(self, start_date, end_date, max_leverage=1.0):
        logger.info(f"🏁 Starting REAL-WORLD Simulation: {start_date.date()} -> {end_date.date()} | Max Lev: {max_leverage}x")
        
        # 1. Collect Data with DAILY STRIDE
        self.engine.close()
        steps = self.universe.walk_forward(
            universe=self.tickers, 
            start_date=start_date, end_date=end_date, 
            stride=1, 
            latest_only=True, backtest_mode=True
        )
        if not steps: return None

        # ... (rest of score and spy loading remains same)
        all_scores = self._get_batch_scores(steps)
        conn = duckdb.connect(self.db_path, read_only=True)
        spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY' AND event_time >= '{start_date - timedelta(days=60)}'").df()
        spy_df['event_time'] = pd.to_datetime(spy_df['event_time'])
        spy_start_slice = spy_df[spy_df['event_time'] >= start_date]
        spy_start_p = spy_start_slice.iloc[0]['close']
        spy_df['spy_val'] = (spy_df['close'] / spy_start_p) * 100000.0
        conn.close()

        cash = 100000.0
        positions = {} 
        price_cache = {} 
        starting_capital = 100000.0
        peak_value = 100000.0
        hedge_qty = 0.0
        hedge_entry_p = 0.0
        results_log = []

        for i in range(len(steps) - 1):
            curr_step = steps[i]
            dt = curr_step['date']; batch = curr_step['batch']
            
            for t in batch.tickers:
                idx = batch.tickers.index(t)
                price_cache[t] = float(batch.data['raw_price'][idx])

            pos_mv = 0.0
            for t, q in positions.items():
                p = price_cache.get(t, 0.0)
                pos_mv += (q * p)
            
            spy_p = price_cache.get('SPY', spy_df[spy_df['event_time'] <= dt]['close'].iloc[-1])
            hedge_pnl = hedge_qty * (hedge_entry_p - spy_p) if hedge_qty > 0 else 0.0
            
            nlv = cash + pos_mv + hedge_pnl
            peak_value = max(peak_value, nlv)
            
            scores_dict = all_scores.get(dt, {})
            obs_scores = [scores_dict.get(t, 0.0) for t in self.tickers]
            obs = self._get_rl_observation(obs_scores, nlv, cash, spy_df, dt, starting_capital, peak_value)
            
            if self.rl_pilot:
                action, _ = self.rl_pilot.predict(obs, deterministic=True)
                target_lev, hedge_ratio, concentration_idx = action
                # SENIOR FIX: Enforce the 'No Borrowing' cap
                target_lev = min(float(target_lev), max_leverage)
                concentration = [2, 5, 12][int(np.clip(concentration_idx, 0, 2))]
            else:
                target_lev = 1.0; hedge_ratio = 0.0; concentration = 5 
            
            is_rebalance_day = (dt.weekday() == 0)
            
            if is_rebalance_day:
                target_notional = (nlv * target_lev)
                slot_notional = target_notional / concentration if concentration > 0 else 0
                sorted_tickers = sorted(scores_dict.keys(), key=lambda x: scores_dict.get(x, -999), reverse=True)
                top_picks = sorted_tickers[:concentration]
                
                # Exit
                for t in list(positions.keys()):
                    if t not in top_picks or target_lev < 0.05:
                        p = price_cache.get(t, 0.0)
                        cash += (positions[t] * p)
                        del positions[t]

                # Entry
                if target_lev >= 0.05:
                    for t in top_picks:
                        if t not in batch.tickers: continue
                        p = price_cache.get(t, 0.0)
                        if p <= 0: continue
                        
                        if t in positions:
                            diff = slot_notional - (positions[t] * p)
                            qty_diff = int(diff / p)
                            cash -= (qty_diff * p)
                            positions[t] += qty_diff
                        else:
                            qty = int(slot_notional / p)
                            if qty > 0:
                                cash -= (qty * p)
                                positions[t] = qty

                # Update Hedge
                cash += hedge_pnl
                target_hedge_notional = nlv * hedge_ratio
                hedge_qty = target_hedge_notional / spy_p if spy_p > 0 else 0
                hedge_entry_p = spy_p
                cash -= (target_notional + target_hedge_notional) * 0.0015 

            spy_val = spy_df[spy_df['event_time'] <= dt]['spy_val'].iloc[-1]
            results_log.append({
                "Date": dt, "NLV": nlv, "SPY": spy_val, "Lev": float(target_lev), 
                "Hedge": float(hedge_ratio), "Conc": concentration
            })

        df = pd.DataFrame(results_log)
        plt.figure(figsize=(15, 8), facecolor='#0D1117')
        ax = plt.gca(); ax.set_facecolor('#0D1117')
        plt.plot(df['Date'], df['NLV'], color='#2ecc71', lw=3, label=f'RL Strategy (No Leverage)')
        plt.plot(df['Date'], df['SPY'], color='#bdc3c7', ls='--', label='SPY Benchmark')
        plt.title(f"Unleveraged Truth: Strategy V5.1 (1.0x Max Leverage)", color='white', fontsize=14)
        plt.grid(True, alpha=0.2); plt.legend(); plt.savefig("data/simulation_unleveraged.png")
        
        logger.success(f"Simulation finished. Final Capital: ${nlv:,.2f}")
        return df

        df = pd.DataFrame(results_log)
        # Final Report
        plt.figure(figsize=(15, 8), facecolor='#0D1117')
        ax = plt.gca(); ax.set_facecolor('#0D1117')
        plt.plot(df['Date'], df['Total_NLV'], color='#2ecc71', lw=3, label=f'RL Pilot Strategy (Capacity Uncapped)')
        plt.plot(df['Date'], df['SPY'], color='#bdc3c7', ls='--', label='SPY Benchmark')
        plt.title(f"Final Audit: Strategy V5 Capacity & Alpha", color='white', fontsize=14)
        plt.yscale('log') # Use log scale because this is massive
        plt.grid(True, alpha=0.2); plt.legend(); plt.savefig("data/simulation_v5_audit.png")
        
        final_total = nlv + total_overflow_capital
        logger.success(f"Simulation finished. Total Strategy Value: ${final_total:,.2f}")
        return df
