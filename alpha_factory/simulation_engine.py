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

class SimulationEngineV5:
    """
    Expert-Grade Simulation Engine (Ferrari Edition).
    - Batch Inference for speed.
    - Audited Institutional Ledger.
    - SPY Benchmarking.
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

    def run(self, start_date, end_date, leverage=2.0, concentration=5, threshold=0.15):
        logger.info(f"🏁 Starting Simulation: {start_date.date()} -> {end_date.date()} | Lev: {leverage}x | Top: {concentration}")
        
        # 1. Collect Daily Data
        self.engine.close()
        steps = self.universe.walk_forward(
            universe=self.tickers, 
            start_date=start_date, end_date=end_date, 
            stride=1, 
            latest_only=True, backtest_mode=True
        )
        if not steps: return None

        # 2. Batch AI Logic
        all_scores = self._get_batch_scores(steps)
        
        # 3. SPY Benchmark
        conn = duckdb.connect(self.db_path, read_only=True)
        spy_df = conn.execute(f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY' AND event_time >= '{start_date}'").df()
        spy_df['event_time'] = pd.to_datetime(spy_df['event_time'])
        spy_start_p = spy_df.iloc[0]['close']
        spy_df['spy_val'] = (spy_df['close'] / spy_start_p) * 100000.0
        conn.close()

        # 4. Trading Loop
        cash = 100000.0
        positions = {} 
        avg_costs = {}
        belief = 0.5
        results_log = []

        for i in range(len(steps) - 1):
            curr_step = steps[i]; next_step = steps[i+1]
            dt = curr_step['date']; batch = curr_step['batch']
            
            # Ledger NLV
            pos_mv = 0.0
            for t, q in positions.items():
                p = float(batch.data['raw_price'][batch.tickers.index(t)]) if t in batch.tickers else avg_costs[t]
                pos_mv += (q * p)
            nlv = cash + pos_mv
            
            # Belief update proxy
            # In V6 this will be replaced by the RL Pilot
            belief = 0.95 if (nlv > 100000) else max(0.05, belief - 0.01)
            
            # Logic
            is_active = belief > threshold
            target_notional = (nlv * leverage) if is_active else 0.0
            slot_notional = target_notional / concentration
            
            is_rebalance_day = (dt.weekday() == 0)
            scores = all_scores.get(dt, {})
            sorted_tickers = sorted(scores.keys(), key=lambda x: scores.get(x, -999), reverse=True)
            
            # Exit
            top_hold = sorted_tickers[:concentration*2]
            for t in list(positions.keys()):
                if (t not in top_hold or not is_active) and is_rebalance_day:
                    p = float(batch.data['raw_price'][batch.tickers.index(t)]) if t in batch.tickers else avg_costs[t]
                    cash += (positions[t] * p)
                    del positions[t]; del avg_costs[t]

            # Entry
            if is_active and is_rebalance_day:
                for t in sorted_tickers[:concentration]:
                    if t not in batch.tickers: continue
                    p = float(batch.data['raw_price'][batch.tickers.index(t)])
                    if p <= 0: continue
                    
                    if t in positions:
                        diff = slot_notional - (positions[t] * p)
                        if abs(diff) > (slot_notional * 0.2):
                            qty_diff = int(diff / p)
                            cash -= (qty_diff * p)
                            positions[t] += qty_diff
                    else:
                        qty = int(slot_notional / p)
                        if qty > 0:
                            cash -= (qty * p)
                            positions[t] = qty; avg_costs[t] = p

            # Benchmarking
            spy_val = spy_df[spy_df['event_time'] <= dt]['spy_val'].iloc[-1]
            results_log.append({
                "Date": dt, "NLV": nlv, "SPY": spy_val, "Belief": belief, "Pos": len(positions)
            })

        df = pd.DataFrame(results_log)
        # Final Report
        plt.figure(figsize=(15, 8))
        plt.plot(df['Date'], df['NLV'], color='#2ecc71', lw=3, label=f'AI Strategy ({leverage}x)')
        plt.plot(df['Date'], df['SPY'], color='#bdc3c7', ls='--', label='SPY Benchmark')
        plt.title(f"Simulation Outcome: ${nlv:,.2f} vs SPY ${spy_val:,.2f}")
        plt.savefig("data/simulation_outcome.png")
        logger.success(f"Simulation finished. Plot saved to data/simulation_outcome.png")
        return df
