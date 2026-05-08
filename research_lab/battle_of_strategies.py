import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_ranker import MultiModalRankNet, InputSpec
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

class StrategyBattle:
    def __init__(self, tickers=None):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['storage_path']
        self.tickers = tickers if tickers else self.config['universe']['tickers']
        
        # Ensure SPY is in tickers for benchmark
        if "SPY" not in self.tickers:
            self.tickers.append("SPY")
            
        # Initialize AlphaUniverse
        # We'll use a local connection to avoid pickling issues if we were parallelizing, 
        # but for this script we might just run sequentially for simplicity or use the built-in parallel walk_forward.
        import duckdb
        self.conn = duckdb.connect(db_path, read_only=True)
        self.universe = AlphaUniverse(conn=self.conn, config=self.config)
        
        # Load AI Model (Challenger)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self.config['model_pipeline'].get('model_path', 'models/challenger_v2.pt')
        
        lookback = self.config['signal_physics'].get('lookback_days', 63)
        n_scales = len(self.config['signal_physics']['wavelet_transform']['scales'])
        self.specs = [
            InputSpec(name='x_seq', shape=(lookback, 1), type='seq'),
            InputSpec(name='x_spatial', shape=(1, n_scales, lookback), type='spatial'),
            InputSpec(name='x_graph', shape=(8,), type='graph'),
            InputSpec(name='x_volume', shape=(lookback, 1), type='seq')
        ]
        
        try:
            if os.path.exists(model_path):
                self.ai_model = torch.jit.load(model_path).to(self.device)
                self.ai_model.eval()
                logger.info(f"Loaded AI Model from {model_path}")
            else:
                logger.warning(f"AI Model {model_path} not found. Will use random ranking for AI agent as placeholder.")
                self.ai_model = None
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.ai_model = None

    def get_ai_scores(self, batch):
        if self.ai_model is None:
            return np.random.randn(len(batch.tickers))
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in batch.data.items() if k in [s.name for s in self.specs]}
            scores = self.ai_model(inputs).squeeze()
            if self.device.type == 'cuda':
                scores = scores.cpu()
            return scores.numpy()

    def get_momentum_scores(self, batch):
        # 60-day momentum (full lookback in x_seq)
        momentum = batch.data['x_seq'].mean(dim=1).squeeze().numpy()
        return momentum

    def get_mean_reversion_scores(self, batch):
        # Short-term reversal (5-day)
        rev_score = -batch.data['x_seq'][:, -5:].mean(dim=1).squeeze().numpy()
        return rev_score

    def get_low_vol_scores(self, batch):
        # Inverse of realized volatility (std of returns)
        # x_seq is standardized, but its relative std still matters
        vol = torch.std(batch.data['x_seq'], dim=1).squeeze().numpy()
        return -vol # Lower vol = Higher score

    def run_backtest(self, start_date, end_date):
        logger.info(f"Running Battle of Strategies from {start_date.date()} to {end_date.date()}")
        
        # We'll rebalance weekly (stride=5)
        steps = self.universe.walk_forward(
            universe=[t for t in self.tickers if t != "SPY"], 
            start_date=start_date, 
            end_date=end_date, 
            stride=5, 
            latest_only=True,
            backtest_mode=True
        )
        
        # SENIOR FIX: Re-establish connection because walk_forward closes it for parallel safety
        db_path = self.config['data_engine']['storage_path']
        import duckdb
        self.conn = duckdb.connect(db_path, read_only=True)
        self.universe._conn = self.conn

        if not steps:
            logger.error("No data produced for backtest.")
            return

        # Track results
        portfolio_values = {
            'AI_Challenger': 1.0,
            'Momentum_King': 1.0,
            'Mean_Reversion': 1.0,
            'Low_Vol_Sentinel': 1.0,
            'SPY_Benchmark': 1.0
        }
        
        # To calculate SPY performance, we need its returns separately
        spy_query = f"SELECT event_time, close FROM market_data WHERE ticker = 'SPY' AND event_time >= '{start_date}' AND event_time <= '{end_date}' ORDER BY event_time"
        spy_data = self.conn.execute(spy_query).df().set_index('event_time')
        spy_returns = spy_data['close'].pct_change().fillna(0)
        
        history = []
        portfolio_log = []

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i+1]
            
            date = current_step['date']
            batch = current_step['batch']
            
            # 1. Get Scores
            scores = {
                'AI_Challenger': self.get_ai_scores(batch),
                'Momentum_King': self.get_momentum_scores(batch),
                'Mean_Reversion': self.get_mean_reversion_scores(batch),
                'Low_Vol_Sentinel': self.get_low_vol_scores(batch)
            }
            
            # 2. Pick Top 5 for each strategy
            portfolios = {}
            for name, score_arr in scores.items():
                if np.isscalar(score_arr):
                    top_indices = [0]
                else:
                    top_indices = np.argsort(score_arr)[-5:] # Top 5
                portfolios[name] = [batch.tickers[idx] for idx in top_indices]
            
            # 3. Calculate Returns until next rebalance
            next_prices = {t: p for t, p in zip(next_step['batch'].tickers, next_step['batch'].data['raw_price'].numpy())}
            curr_prices = {t: p for t, p in zip(batch.tickers, batch.data['raw_price'].numpy())}
            
            step_returns = {}
            for name, selected_tickers in portfolios.items():
                ret_sum = 0
                count = 0
                for t in selected_tickers:
                    if t in curr_prices and t in next_prices and curr_prices[t] > 0:
                        ret_sum += (next_prices[t] - curr_prices[t]) / curr_prices[t]
                        count += 1
                
                avg_ret = ret_sum / count if count > 0 else 0
                portfolio_values[name] *= (1 + avg_ret)
                step_returns[name] = avg_ret
            
            # Update SPY Benchmark
            spy_slice = spy_returns[(spy_returns.index >= current_step['date']) & (spy_returns.index < next_step['date'])]
            spy_period_ret = (1 + spy_slice).prod() - 1
            portfolio_values['SPY_Benchmark'] *= (1 + spy_period_ret)
            
            history.append({
                'date': date,
                **portfolio_values
            })
            
            if i % 10 == 0 or i == len(steps) - 2:
                logger.info(f"Date: {date.date()} | AI: {portfolio_values['AI_Challenger']:.3f} | SPY: {portfolio_values['SPY_Benchmark']:.3f}")
                portfolio_log.append({'date': date, 'portfolios': portfolios})

        df_results = pd.DataFrame(history)
        df_results.to_csv("data/strategy_battle_long_term.csv", index=False)
        
        # Summary
        print("\n" + "="*60)
        print("BATTLE OF STRATEGIES (2023-2026) - FINAL RESULTS")
        print("="*60)
        print(f"{'Strategy':25} | {'Total Return':15} | {'Max Drawdown':15}")
        print("-" * 60)
        for col in df_results.columns:
            if col == 'date': continue
            
            # Calculate Max Drawdown
            series = df_results[col]
            roll_max = series.cummax()
            drawdown = (series - roll_max) / roll_max
            max_dd = drawdown.min() * 100
            
            final_val = series.iloc[-1]
            total_ret = (final_val - 1) * 100
            print(f"{col:25} | {total_ret:10.2f}% | {max_dd:10.2f}%")
        print("="*60)
        
        # Plotting
        try:
            plt.figure(figsize=(12, 7))
            for col in df_results.columns:
                if col == 'date': continue
                plt.plot(df_results['date'], df_results[col], label=col)
            
            plt.title("Battle of Strategies: 2023 - 2026 Cumulative Returns")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value (1.0 = Start)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("data/strategy_battle_2026.png")
            logger.info("Equity curve plot saved to data/strategy_battle_2026.png")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
        
        # Detailed Log for last rebalance
        print("\nTOP PICKS (Final Rebalance):")
        last_log = portfolio_log[-1]
        for name, picks in last_log['portfolios'].items():
            print(f"{name:20}: {', '.join(picks)}")
        
        return df_results

if __name__ == "__main__":
    battle = StrategyBattle()
    # Long Term Backtest: 2023 to 2026
    results = battle.run_backtest(datetime(2023, 1, 1), datetime(2026, 5, 1))
