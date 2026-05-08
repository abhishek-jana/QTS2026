import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_ranker import MultiModalRankNet, InputSpec
import matplotlib.pyplot as plt

class RadicalStrategyBattle:
    def __init__(self):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['storage_path']
        self.tickers = self.config['universe']['tickers']
        
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

    def run_battle(self, start_date, end_date):
        logger.info(f"🔥 RADICAL STRATEGY BATTLE ({start_date.date()} -> {end_date.date()}) 🔥")
        
        # Load IC for Bayesian Belief
        ic_df = pd.read_csv("data/backtest_results.csv")
        ic_map = {row['date']: row['challenger_ic'] for _, row in ic_df.iterrows()}
        
        steps = self.universe.walk_forward(
            universe=[t for t in self.tickers if t != "SPY"], 
            start_date=start_date, 
            end_date=end_date, 
            stride=5, # Weekly rebalance
            latest_only=True,
            backtest_mode=True
        )
        
        # Re-open DB
        db_path = self.config['data_engine']['storage_path']
        import duckdb
        self.conn = duckdb.connect(db_path, read_only=True)
        self.universe._conn = self.conn

        # Strategy definitions
        # 1. The Sniper: Only Top 2 stocks, high belief threshold (0.80), 100% exposure
        # 2. The Diversifier: Top 15 stocks, risk-parity weight (1/vol), 80% exposure
        # 3. The Contrarian: AI top decile but only if 5-day return is negative (mean reversion overlay)
        # 4. The Regime-Follower: Exposure scales linearly with belief (0.2 to 1.5 leverage)

        p_values = {
            'Sniper_Conviction': 1.0,
            'Risk_Parity_Diversifier': 1.0,
            'AI_Contrarian': 1.0,
            'Regime_Leveraged': 1.0,
            'SPY_Benchmark': 1.0
        }
        
        spy_query = f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY' AND event_time >= '{start_date}' AND event_time <= '{end_date}' ORDER BY event_time"
        spy_data = self.conn.execute(spy_query).df().set_index('event_time')
        spy_returns = spy_data['close'].pct_change().fillna(0)
        
        belief = 0.50
        history = []

        for i in range(len(steps) - 1):
            curr_step = steps[i]
            next_step = steps[i+1]
            date = curr_step['date']
            batch = curr_step['batch']
            
            # Update Belief
            ic = ic_map.get(date.strftime("%Y-%m-%d"), 0.0)
            l_v = 1 / (1 + np.exp(-5 * (ic - 0.01)))
            marginal = (l_v * belief) + ((1-l_v) * (1 - belief))
            if marginal > 0: belief = (l_v * belief) / marginal
            belief = max(0.05, min(0.95, belief))
            
            scores = self.get_ai_scores(batch)
            next_prices = {t: p for t, p in zip(next_step['batch'].tickers, next_step['batch'].data['raw_price'].numpy())}
            curr_prices = {t: p for t, p in zip(batch.tickers, batch.data['raw_price'].numpy())}
            
            def calc_ret(selected_tickers, weights=None):
                if not selected_tickers: return 0.0
                rets = []
                for t in selected_tickers:
                    if t in curr_prices and t in next_prices and curr_prices[t] > 0:
                        rets.append((next_prices[t] - curr_prices[t]) / curr_prices[t])
                    else: rets.append(0.0)
                if weights is None: return np.mean(rets)
                return np.sum(np.array(rets) * np.array(weights))

            # --- 1. The Sniper ---
            if belief > 0.80:
                sniper_picks = [batch.tickers[idx] for idx in np.argsort(scores)[-2:]]
                p_values['Sniper_Conviction'] *= (1.0 + calc_ret(sniper_picks))
            
            # --- 2. Risk Parity Diversifier ---
            if belief > 0.60:
                div_indices = np.argsort(scores)[-15:]
                div_tickers = [batch.tickers[idx] for idx in div_indices]
                # Inverse vol weighting (proxy using x_seq std)
                vols = torch.std(batch.data['x_seq'][div_indices], dim=1).squeeze().numpy()
                inv_vols = 1.0 / (vols + 1e-6)
                weights = inv_vols / inv_vols.sum()
                p_values['Risk_Parity_Diversifier'] *= (1.0 + calc_ret(div_tickers, weights) * 0.80)
            
            # --- 3. AI Contrarian ---
            if belief > 0.65:
                top_decile_idx = np.argsort(scores)[-10:]
                # Check recent 5-day return from x_seq
                recent_rets = batch.data['x_seq'][top_decile_idx, -5:].mean(dim=1).squeeze().numpy()
                contrarian_idx = top_decile_idx[recent_rets < 0]
                if len(contrarian_idx) > 0:
                    contrarian_tickers = [batch.tickers[idx] for idx in contrarian_idx]
                    p_values['AI_Contrarian'] *= (1.0 + calc_ret(contrarian_tickers))
            
            # --- 4. Regime Leveraged ---
            # Leverage = belief * 1.5 (max 1.5, min 0.1)
            leverage = belief * 1.5
            regime_picks = [batch.tickers[idx] for idx in np.argsort(scores)[-5:]]
            p_values['Regime_Leveraged'] *= (1.0 + calc_ret(regime_picks) * leverage)

            # Benchmark
            mask = (spy_returns.index >= date) & (spy_returns.index < next_step['date'])
            spy_slice = spy_returns[mask]
            p_values['SPY_Benchmark'] *= (1.0 + ((1 + spy_slice).prod() - 1))
            
            history.append({'date': date, **p_values})

        df = pd.DataFrame(history)
        print("\n" + "="*80)
        print(f"{'Strategy':30} | {'Return %':>12} | {'Max DD %':>12} | {'Sharpe':>8}")
        print("-" * 80)
        for col in df.columns:
            if col == 'date': continue
            ret = (df[col].iloc[-1] - 1) * 100
            dd = ((df[col] - df[col].cummax()) / df[col].cummax()).min() * 100
            # Rough Sharpe
            daily_rets = df[col].pct_change().dropna()
            sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(52) if daily_rets.std() > 0 else 0
            print(f"{col:30} | {ret:11.2f}% | {dd:11.2f}% | {sharpe:8.2f}")
        print("="*80 + "\n")
        
        return df

if __name__ == "__main__":
    battle = RadicalStrategyBattle()
    battle.run_battle(datetime(2023, 1, 1), datetime(2026, 5, 1))
