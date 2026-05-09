import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import duckdb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# SENIOR FIX: Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

class HighPerformanceProductionSim:
    """
    Expert-Grade Simulation Engine.
    Features: Batch Inference, Professional Ledger V4, SPY Benchmarking, and Matplotlib Plotting.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        db_path = self.config['data_engine']['storage_path']
        
        logger.info(f"HP Sim: Loading universe data from {db_path} into RAM...")
        disk_conn = duckdb.connect(db_path, read_only=True)
        
        # Load EVERYTHING needed into a dataframe
        self.all_data_df = disk_conn.execute("SELECT * FROM market_data").df()
        disk_conn.close()
        
        # Setup Memory DuckDB for the strategy engine
        self.mem_conn = duckdb.connect(":memory:")
        self.mem_conn.register("market_data", self.all_data_df)
        
        class MemoryDataEngine:
            def __init__(self, conn): self.conn = conn
            def get_pit_view(self, ticker, as_of):
                query = f"SELECT * FROM market_data WHERE ticker = '{ticker}' AND event_time <= '{as_of}' ORDER BY event_time ASC"
                return self.conn.execute(query).df().set_index('event_time')

        self.data_engine = MemoryDataEngine(self.mem_conn)
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        self.strategy.lab._conn = self.mem_conn
        
        # Simulation State
        self.starting_capital = 100000.0
        self.sim_cash = 100000.0
        self.sim_positions = {} 
        self.sim_avg_costs = {}
        self.ranking_history = [] 

    def _get_latest_price(self, ticker, current_time):
        try:
            # High speed dataframe slice
            subset = self.all_data_df[(self.all_data_df['ticker'] == ticker) & (self.all_data_df['event_time'] <= current_time)]
            if not subset.empty: return float(subset.iloc[-1]['close'])
        except: pass
        return 0.0

    def _get_market_regime(self, current_time):
        try:
            spy_data = self.all_data_df[(self.all_data_df['ticker'] == 'SPY') & (self.all_data_df['event_time'] <= current_time)]
            if len(spy_data) > 21:
                p_now = float(spy_data.iloc[-1]['close'])
                p_past = float(spy_data.iloc[-22]['close'])
                return "BULL" if p_now > p_past else "BEAR"
        except: pass
        return "BULL"

    def _update_metacognition(self, current_time):
        horizon_days = self.config.get('signal_physics', {}).get('horizon_days', 5)
        target_time = current_time - timedelta(days=horizon_days)
        
        eval_item = None
        for item in self.ranking_history:
            if item['time'] <= target_time:
                eval_item = item
            else: break
        
        if not eval_item: return
        self.ranking_history = [item for item in self.ranking_history if item['time'] >= target_time - timedelta(days=2)]
        
        realized_returns = {}
        for ticker in self.tickers:
            p0 = self._get_latest_price(ticker, eval_item['time'])
            p1 = self._get_latest_price(ticker, current_time)
            if p0 > 0: realized_returns[ticker] = (p1 / p0) - 1.0
        
        if realized_returns:
            self.strategy.update_model_metacognition(realized_returns, eval_item['rankings'])

    def run(self, start_date, end_date):
        logger.info(f"🚀 OPTIMIZED BATCH SIM: {start_date.date()} -> {end_date.date()}")
        
        # 1. PRE-COMPUTE ALL SPY PRICES FOR BENCHMARK
        spy_df = self.all_data_df[self.all_data_df['ticker'] == 'SPY'].copy().sort_values('event_time')
        spy_df = spy_df[(spy_df['event_time'] >= start_date) & (spy_df['event_time'] <= end_date)]
        if not spy_df.empty:
            spy_start_p = spy_df.iloc[0]['close']
            spy_df['benchmark_val'] = (spy_df['close'] / spy_start_p) * self.starting_capital

        current_time = start_date
        results = []
        
        while current_time <= end_date:
            self._update_metacognition(current_time)
            
            view = self.strategy.get_current_rankings(as_of=current_time)
            if view['status'] != "OK":
                current_time += timedelta(days=1)
                continue
                
            belief = float(view['belief_score'])
            self.ranking_history.append({'time': current_time, 'rankings': {e['ticker']: e['score'] for e in view['ladder']}})
            
            # 2. AUDITED LEDGER: Net Liquidation Value
            # NLV = Cash + Sum(Market Value of Positions)
            pos_market_value = 0.0
            for t, qty in self.sim_positions.items():
                p = self._get_latest_price(t, current_time) or self.sim_avg_costs[t]
                pos_market_value += (qty * p)
            
            nlv = self.sim_cash + pos_market_value
            
            # 3. AUDITED MATH: Sizing & Concentration
            # High-Octane Flagship: 2.0x Fixed Leverage, Top 5
            threshold = 0.15
            is_active = belief > threshold
            leverage = 2.0 if is_active else 0.0
            target_total_notional = nlv * leverage
            notional_per_slot = target_total_notional / 5
            
            # 4. Weekly Rebalance Cadence
            is_rebalance_day = (current_time.weekday() == 0)
            
            # Exit Logic
            hold_size = 8
            top_hold = [e['ticker'] for e in view['ladder'][:hold_size]]
            
            for t in list(self.sim_positions.keys()):
                qty = self.sim_positions[t]
                should_close = (t not in top_hold) or not is_active
                if should_close and (is_rebalance_day or nlv < self.starting_capital * 0.7):
                    p = self._get_latest_price(t, current_time)
                    if p > 0:
                        self.sim_cash += (qty * p)
                        del self.sim_positions[t]; del self.sim_avg_costs[t]

            # Entry Logic
            if is_active and is_rebalance_day:
                top_entry = [e['ticker'] for e in view['ladder'][:5]]
                for t in top_entry:
                    p = self._get_latest_price(t, current_time)
                    if p <= 0: continue
                    
                    if t in self.sim_positions:
                        # AUDITED REBALANCING: Weighted Average Cost & Dynamic Trimming
                        cur_val = self.sim_positions[t] * p
                        diff = notional_per_slot - cur_val
                        if abs(diff) > (notional_per_slot * 0.2):
                            qty_diff = int(diff / p)
                            if qty_diff != 0:
                                self.sim_cash -= (qty_diff * p)
                                self.sim_positions[t] += qty_diff
                                # Update avg cost if adding
                                if qty_diff > 0:
                                    self.sim_avg_costs[t] = ((self.sim_avg_costs[t] * (self.sim_positions[t]-qty_diff)) + (p * qty_diff)) / self.sim_positions[t]
                    else:
                        qty = int(notional_per_slot / p)
                        if qty > 0:
                            self.sim_cash -= (qty * p)
                            self.sim_positions[t] = qty
                            self.sim_avg_costs[t] = p

            # 5. Benchmarking
            spy_slice = spy_df[spy_df['event_time'] <= current_time]
            spy_val = spy_slice['benchmark_val'].iloc[-1] if not spy_slice.empty else self.starting_capital

            results.append({
                "Date": current_time,
                "Account_Value": nlv,
                "SPY_Benchmark": spy_val,
                "Belief": belief,
                "Positions": len(self.sim_positions),
                "Gross_Exp": (abs(pos_market_value) / nlv * 100) if nlv > 0 else 0
            })
            
            if current_time.day == 1:
                logger.info(f"{current_time.date()} | NLV: ${nlv:,.0f} | SPY: ${spy_val:,.0f} | Belief: {belief:.2%}")
                
            current_time += timedelta(days=1)
            if current_time.day % 7 == 0:
                torch.cuda.empty_cache()
                self.strategy.lab._feat_cache = {}

        df = pd.DataFrame(results)
        df.to_csv("data/high_performance_audit_results.csv", index=False)
        
        # 6. PROFESSIONAL PLOTTING
        try:
            plt.figure(figsize=(14, 8))
            plt.plot(df['Date'], df['Account_Value'], label='High-Octane AI (2.0x)', color='#2ecc71', linewidth=2.5)
            plt.plot(df['Date'], df['SPY_Benchmark'], label='S&P 500 Benchmark', color='#bdc3c7', linestyle='--', alpha=0.8)
            
            plt.fill_between(df['Date'], df['Account_Value'], df['SPY_Benchmark'], 
                             where=(df['Account_Value'] > df['SPY_Benchmark']), 
                             color='#2ecc71', alpha=0.1, interpolate=True, label='Alpha Generation')
            
            plt.title(f"High-Octane AI Performance vs. S&P 500 (2023-2026)\nFinal NLV: ${nlv:,.2f}", fontsize=14, fontweight='bold')
            plt.xlabel("Date")
            plt.ylabel("Account Value ($)")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig("data/high_octane_vs_spy.png")
            logger.success("📊 Performance plot saved to data/high_octane_vs_spy.png")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

        logger.success(f"Final Return: {((nlv/self.starting_capital)-1)*100:.2f}% | SPY Return: {((spy_val/self.starting_capital)-1)*100:.2f}%")
        return df

if __name__ == "__main__":
    sim = HighPerformanceProductionSim()
    sim.run(datetime(2023, 1, 1), datetime(2026, 5, 1))
