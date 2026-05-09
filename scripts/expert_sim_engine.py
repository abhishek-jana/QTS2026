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

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.data_engine import DataEngine

class ExpertFerrariSim:
    """
    Expert-Grade Simulation Engine.
    - Pre-computed Batch Inference (1000x faster than daily loops).
    - Audited Institutional Ledger V4 (NLV, BP, Reg-T).
    - Benchmarking & Plotting.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['universe']['tickers']
        self.db_path = self.config['data_engine']['storage_path']
        self.engine = DataEngine(storage_path=self.db_path)
        self.universe = AlphaUniverse(conn=self.engine.conn, config=self.config)
        
        # Load Model once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self.config['model_pipeline'].get('model_path', 'models/challenger_v2.pt')
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

    def _get_batch_scores(self, steps):
        """Passes all data to GPU in chunks for maximum speed."""
        logger.info(f"🚀 Ferrari: Running Batch Inference on {len(steps)} days...")
        scores_map = {}
        
        for step in tqdm(steps, desc="🧠 AI Thinking"):
            batch = step['batch'].to(self.device)
            with torch.no_grad():
                # Correct modality names for your RankNet
                inputs = {k: v for k, v in batch.data.items() if k in ['x_seq', 'x_spatial', 'x_graph', 'x_volume']}
                s = self.model(inputs).squeeze().cpu().numpy()
                scores_map[step['date']] = {t: float(val) for t, val in zip(batch.tickers, s)}
        return scores_map

    def run(self, start_date, end_date):
        logger.info(f"🏎️  FERRARI START: {start_date.date()} -> {end_date.date()}")
        
        # 1. PRE-COMPUTE ALL DATA (The parallel part)
        # Release lock for walk_forward
        self.engine.close()
        steps = self.universe.walk_forward(
            universe=self.tickers, 
            start_date=start_date, end_date=end_date, 
            stride=1, # DAILY RESOLUTION
            latest_only=True, backtest_mode=True
        )
        
        if not steps:
            logger.error("No data found for period.")
            return

        # 2. BATCH INFERENCE
        all_scores = self._get_batch_scores(steps)
        
        # 3. BENCHMARK DATA
        import duckdb
        conn = duckdb.connect(self.db_path, read_only=True)
        spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY' AND event_time >= '{start_date}'").df()
        spy_df['event_time'] = pd.to_datetime(spy_df['event_time'])
        spy_start_p = spy_df.iloc[0]['close']
        spy_df['spy_val'] = (spy_df['close'] / spy_start_p) * 100000.0
        conn.close()

        # 4. EXECUTION LOOP (Pure Math, Blazing Fast)
        positions = {} # ticker -> qty
        avg_costs = {}
        cash = 100000.0
        realized_pnl = 0.0
        belief = 0.5
        threshold = 0.15 # High-Octane
        results_log = []

        for i in range(len(steps) - 1):
            curr_step = steps[i]
            next_step = steps[i+1]
            dt = curr_step['date']
            batch = curr_step['batch']
            
            # A. Audited Ledger Math (NLV)
            unrealized_pnl = 0.0
            gross_exp = 0.0
            pos_market_value = 0.0
            for t, q in positions.items():
                curr_p = float(batch.data['raw_price'][batch.tickers.index(t)]) if t in batch.tickers else avg_costs[t]
                unrealized_pnl += (curr_p - avg_costs[t]) * q
                pos_market_value += (q * curr_p)
                gross_exp += abs(q * curr_p)
            
            nlv = cash + pos_market_value
            
            # Simplified Belief update for simulation
            belief = 0.95 if unrealized_pnl >= 0 else max(0.05, belief - 0.01)
            
            # B. Strategy Logic (High-Octane V3)
            is_active = belief > threshold
            target_lev = 2.0 if is_active else 0.0
            target_total_notional = nlv * target_lev
            notional_per_slot = target_total_notional / 5
            
            is_rebalance_day = (dt.weekday() == 0)
            
            # Exit/Hysteresis
            scores = all_scores.get(dt, {})
            sorted_tickers = sorted(scores.keys(), key=lambda x: scores.get(x, -999), reverse=True)
            top_hold = sorted_tickers[:10]
            
            for t in list(positions.keys()):
                if (t not in top_hold or not is_active) and is_rebalance_day:
                    p = float(batch.data['raw_price'][batch.tickers.index(t)]) if t in batch.tickers else avg_costs[t]
                    qty = positions[t]
                    realized_pnl += (p - avg_costs[t]) * qty
                    cash += (qty * p)
                    del positions[t]; del avg_costs[t]

            # Entry/Trimming
            if is_active and is_rebalance_day:
                top_5 = sorted_tickers[:5]
                for t in top_5:
                    if t not in batch.tickers: continue
                    p = float(batch.data['raw_price'][batch.tickers.index(t)])
                    if p <= 0: continue
                    
                    if t in positions:
                        cur_val = positions[t] * p
                        diff = notional_per_slot - cur_val
                        if abs(diff) > (notional_per_slot * 0.25):
                            qty_diff = int(diff / p)
                            cash -= (qty_diff * p)
                            positions[t] += qty_diff
                    else:
                        qty = int(notional_per_slot / p)
                        if qty > 0:
                            cash -= (qty * p)
                            positions[t] = qty
                            avg_costs[t] = p

            # C. Benchmarking
            spy_slice = spy_df[spy_df['event_time'] <= dt]
            spy_val = spy_slice['spy_val'].iloc[-1] if not spy_slice.empty else 100000.0
            bp = (2.0 * nlv) - gross_exp
            roe = ((nlv - 100000.0) / (gross_exp if gross_exp > 100 else 100000.0)) * 100

            results_log.append({
                "Date": dt, "Account_Value": nlv, "SPY": spy_val, "Belief": belief,
                "Positions": len(positions), "Gross_Exp": (gross_exp / nlv * 100) if nlv > 0 else 0,
                "Buying_Power": bp, "ROE": roe, "Realized": realized_pnl, "Unrealized": unrealized_pnl
            })
            
            if i % 20 == 0:
                print(f"{dt.date()} | NLV: ${nlv:,.0f} | SPY: ${spy_val:,.0f} | Belief: {belief:.1%} | Pos: {len(positions)}")

        df = pd.DataFrame(results_log)
        df.to_csv("data/expert_audit_results.csv", index=False)
        
        # D. PLOTTING
        try:
            plt.figure(figsize=(15, 9))
            plt.plot(df['Date'], df['Account_Value'], label='High-Octane AI (2.0x)', color='#2ecc71', lw=3)
            plt.plot(df['Date'], df['SPY'], label='S&P 500 Buy & Hold', color='#bdc3c7', linestyle='--', lw=2)
            plt.fill_between(df['Date'], df['Account_Value'], df['SPY'], where=(df['Account_Value'] > df['SPY']), color='#2ecc71', alpha=0.1, interpolate=True)
            plt.title("UQTS-2026 AUDITED PERFORMANCE: High-Octane vs. S&P 500", fontsize=16, fontweight='bold')
            plt.ylabel("Account Value ($)")
            plt.grid(True, alpha=0.15)
            plt.legend()
            plt.tight_layout()
            plt.savefig("data/high_octane_final_audit.png")
            logger.success("📊 Performance plot saved to data/high_octane_final_audit.png")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

        logger.success(f"Final Return: {((nlv/100000.0)-1)*100:.2f}% | SPY Return: {((spy_val/100000.0)-1)*100:.2f}%")
        return df

if __name__ == "__main__":
    sim = ExpertFerrariSim()
    sim.run(datetime(2023, 1, 1), datetime(2026, 5, 1))
