import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_ranker import MultiModalRankNet, InputSpec

class ProBacktester:
    def __init__(self, exposure=0.60):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['storage_path']
        self.tickers = self.config['universe']['tickers']
        self.exposure = exposure
        
        import duckdb
        self.conn = duckdb.connect(db_path, read_only=True)
        self.universe = AlphaUniverse(conn=self.conn, config=self.config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self.config['model_pipeline'].get('model_path', 'models/challenger_v2.pt')
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

    def get_ai_scores(self, batch):
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in batch.data.items() if k in ['x_seq', 'x_spatial', 'x_graph', 'x_volume']}
            scores = self.model(inputs).squeeze()
            return scores.cpu().numpy()

    def run_simulation(self, start_date, end_date):
        logger.info(f"🚀 HIGH-FIDELITY SIMULATION (PRO: {self.exposure*100}% EXPOSURE) 🚀")
        
        # Load IC results for Bayesian Belief update
        ic_df = pd.read_csv("data/backtest_results.csv")
        ic_map = {row['date']: row['challenger_ic'] for _, row in ic_df.iterrows()}
        
        steps = self.universe.walk_forward(
            universe=[t for t in self.tickers if t != "SPY"], 
            start_date=start_date, 
            end_date=end_date, 
            stride=21, 
            latest_only=True,
            backtest_mode=True
        )
        
        # Re-open DB
        db_path = self.config['data_engine']['storage_path']
        import duckdb
        self.conn = duckdb.connect(db_path, read_only=True)
        self.universe._conn = self.conn

        capital = 100000.0
        belief = 0.50
        threshold = 0.65
        history = []
        
        for i in range(len(steps) - 1):
            curr_step = steps[i]
            next_step = steps[i+1]
            date_str = curr_step['date'].strftime("%Y-%m-%d")
            
            # 1. Update Bayesian Belief
            ic = ic_map.get(date_str, 0.0)
            l_v = 1 / (1 + np.exp(-5 * (ic - 0.01)))
            marginal = (l_v * belief) + ((1-l_v) * (1 - belief))
            if marginal > 0: belief = (l_v * belief) / marginal
            belief = max(0.05, min(0.95, belief))
            
            # 2. Execution Logic
            if belief > threshold:
                scores = self.get_ai_scores(curr_step['batch'])
                # Top 4 positions (15% each = 60% exposure)
                top_idx = np.argsort(scores)[-4:]
                selected = [curr_step['batch'].tickers[idx] for idx in top_idx]
                
                # Calculate return
                next_prices = {t: p for t, p in zip(next_step['batch'].tickers, next_step['batch'].data['raw_price'].numpy())}
                curr_prices = {t: p for t, p in zip(curr_step['batch'].tickers, curr_step['batch'].data['raw_price'].numpy())}
                
                rets = []
                for t in selected:
                    if t in curr_prices and t in next_prices and curr_prices[t] > 0:
                        rets.append((next_prices[t] - curr_prices[t]) / curr_prices[t])
                
                avg_ret = np.mean(rets) if rets else 0.0
                capital *= (1.0 + avg_ret * self.exposure)
            
            history.append({'date': curr_step['date'], 'capital': capital, 'belief': belief})
            
        final_ret = (capital / 100000.0 - 1) * 100
        print("\n" + "="*40)
        print("🏁 PRO SIMULATION COMPLETE 🏁")
        print(f"Final Account Value: ${capital:,.2f}")
        print(f"Total Return: {final_ret:.2f}%")
        print(f"Final Bayesian Belief: {belief*100:.1f}%")
        print("="*40 + "\n")

if __name__ == "__main__":
    tester = ProBacktester(exposure=0.60)
    tester.run_simulation(datetime(2023, 1, 1), datetime(2026, 5, 1))
