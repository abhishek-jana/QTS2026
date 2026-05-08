import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import duckdb
from datetime import datetime, timedelta

# SENIOR FIX: Ensure project root is in path for module discovery
sys.path.append(os.getcwd())

from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

class HighPerformanceProductionSim:
    """
    Optimized Simulation Engine.
    Uses In-Memory DuckDB to eliminate Disk I/O and exponentially speed up processing.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        # 1. Open disk DB to fetch data once
        db_path = self.config['data_engine']['storage_path']
        logger.info(f"HP Sim: Loading universe data from {db_path} into RAM...")
        disk_conn = duckdb.connect(db_path, read_only=True)
        
        # Fetch data needed (start - 150 days to end)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2026, 5, 1)
        fetch_start = (start_date - timedelta(days=150)).strftime("%Y-%m-%d")
        
        # Load EVERYTHING into a dataframe
        self.all_data_df = disk_conn.execute(f"SELECT * FROM market_data WHERE event_time >= '{fetch_start}'").df()
        disk_conn.close()
        logger.info(f"HP Sim: Loaded {len(self.all_data_df):,} rows into memory.")

        # 2. Setup Memory-Only DuckDB
        self.mem_conn = duckdb.connect(":memory:")
        self.mem_conn.register("market_data", self.all_data_df)
        
        # 3. Initialize Strategy with Memory Connection
        class MemoryDataEngine:
            def __init__(self, conn): self.conn = conn
            def get_pit_view(self, ticker, as_of):
                query = f"SELECT * FROM market_data WHERE ticker = '{ticker}' AND event_time <= '{as_of}' ORDER BY event_time ASC"
                return self.conn.execute(query).df().set_index('event_time')

        self.data_engine = MemoryDataEngine(self.mem_conn)
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        # Force the strategy's lab to use the memory connection
        self.strategy.lab._conn = self.mem_conn
        
        # Simulation State
        self.starting_capital = 100000.0
        self.sim_cash = 100000.0
        self.sim_positions = {} 
        self.sim_avg_costs = {}
        self.sim_realized_pnl = 0.0
        self.oms_queue = {"filled": 0}
        self.ranking_history = [] # For Bayesian Metacognition

    def _update_metacognition(self, current_time):
        """Learns from realized returns over the target horizon to update Belief."""
        horizon_days = self.config.get('signal_physics', {}).get('horizon_days', 5)
        target_time = current_time - timedelta(days=horizon_days)
        
        eval_item = None
        for item in self.ranking_history:
            if item['time'] <= target_time:
                eval_item = item
            else:
                break
        
        if not eval_item: return
        
        # Clean history
        self.ranking_history = [item for item in self.ranking_history if item['time'] >= target_time - timedelta(days=2)]
        
        realized_returns = {}
        for ticker in self.tickers:
            p0 = self._get_latest_price(ticker, eval_item['time'])
            p1 = self._get_latest_price(ticker, current_time)
            if p0 > 0:
                realized_returns[ticker] = (p1 / p0) - 1.0
        
        if realized_returns:
            self.strategy.update_model_metacognition(realized_returns, eval_item['rankings'])
        
    def _get_latest_price(self, ticker, current_time):
        try:
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

    def run(self, start_date, end_date):
        logger.info(f"🚀 HIGH-OCTANE HP SIM START: {start_date.date()} -> {end_date.date()}")
        
        current_time = start_date
        results = []
        
        while current_time <= end_date:
            # 0. UPDATE METACOGNITION
            self._update_metacognition(current_time)

            # 1. Inference View
            view = self.strategy.get_current_rankings(as_of=current_time)
            if view['status'] != "OK":
                current_time += timedelta(days=1)
                continue
                
            belief = float(view['belief_score'])
            
            # Store ranking
            self.ranking_history.append({
                'time': current_time,
                'rankings': {e['ticker']: e['score'] for e in view['ladder']}
            })
            
            # 2. Update NLV
            unrealized_pnl = 0.0
            gross_notional = 0.0
            for t, qty in self.sim_positions.items():
                p = self._get_latest_price(t, current_time) or self.sim_avg_costs[t]
                unrealized_pnl += (p - self.sim_avg_costs[t]) * qty
                gross_notional += abs(qty * p)
                
            nlv = self.starting_capital + self.sim_realized_pnl + unrealized_pnl
            
            # 3. HIGH-OCTANE SIZING
            # Target = Fixed 2.0x Leverage if belief is alive
            threshold = 0.15 
            leverage = 2.0 if belief > threshold else 0.0
            target_total_notional = nlv * leverage
            
            # HIGH CONCENTRATION: Top 5 slots
            notional_per_slot = target_total_notional / 5
            
            is_rebalance_day = (current_time.weekday() == 0)
            
            total_tickers = len(view['ladder'])
            hold_size = 8 # Ultra tight hold
            top_hold = [e['ticker'] for e in view['ladder'][:hold_size]]

            # 4. Exit (Monday or Extreme Drawdown)
            for t in list(self.sim_positions.keys()):
                qty = self.sim_positions[t]
                should_close = (t not in top_hold) or (belief <= threshold)
                
                if should_close and (is_rebalance_day or nlv < self.starting_capital * 0.7):
                    p = self._get_latest_price(t, current_time)
                    if p > 0:
                        self.sim_realized_pnl += (p - self.sim_avg_costs[t]) * qty
                        self.sim_cash += (qty * p)
                        del self.sim_positions[t]; del self.sim_avg_costs[t]
                        self.oms_queue["filled"] += 1

            # 5. Entry (Monday Only)
            if belief > threshold and is_rebalance_day:
                top_entry = [e['ticker'] for e in view['ladder'][:5]] # Top 5 only
                
                for t in top_entry:
                    p = self._get_latest_price(t, current_time)
                    if p <= 0: continue
                    
                    if t in self.sim_positions:
                        # Dynamic Trimming
                        cur_notional = abs(self.sim_positions[t] * p)
                        # Only rebalance if > 20% deviation
                        if cur_notional > (notional_per_slot * 1.20) or cur_notional < (notional_per_slot * 0.80):
                            diff = cur_notional - notional_per_slot
                            trim_qty = int(abs(diff) / p)
                            if trim_qty > 0:
                                # Update realized pnl and cash
                                trade_pnl = (p - self.sim_avg_costs[t]) * trim_qty if diff > 0 else 0
                                # Simplified partial realized pnl
                                self.sim_realized_pnl += trade_pnl
                                
                                if diff > 0: # Trimming down
                                    self.sim_cash += (trim_qty * p)
                                    self.sim_positions[t] -= trim_qty
                                else: # Buying up
                                    self.sim_cash -= (trim_qty * p)
                                    self.sim_positions[t] += trim_qty
                                self.oms_queue["filled"] += 1
                    else:
                        # New Entry
                        target = notional_per_slot
                        # Use margin (allow BP to go negative up to NLV * 1.0)
                        if self.sim_cash < -nlv: target = 0
                        
                        qty = int(target / p)
                        if qty > 0:
                            self.sim_cash -= (qty * p)
                            self.sim_positions[t] = qty
                            self.sim_avg_costs[t] = p
                            self.oms_queue["filled"] += 1

            # Record Daily Stats
            daily_metrics = {
                "Date": current_time.strftime("%Y-%m-%d"),
                "Active Pos": len(self.sim_positions),
                "Gross Exp (%)": (gross_notional / nlv * 100) if nlv > 0 else 0,
                "Account Value": nlv,
                "Buying Power": self.sim_cash,
                "ROE (Invested)": (self.sim_realized_pnl + unrealized_pnl) / (gross_notional if gross_notional > 100 else 100000.0) * 100,
                "Realized PnL": self.sim_realized_pnl,
                "Unrealized PnL": unrealized_pnl,
                "Bayesian Belief": belief,
                "OMS Filled": self.oms_queue["filled"]
            }
            results.append(daily_metrics)
            
            # Daily Metric Print
            print(f"{daily_metrics['Date']} | Pos: {daily_metrics['Active Pos']:2} | Exp: {daily_metrics['Gross Exp (%)']:6.1f}% | "
                  f"Val: ${daily_metrics['Account Value']:,.2f} | BP: ${daily_metrics['Buying Power']:,.2f} | "
                  f"ROE: {daily_metrics['ROE (Invested)']:6.2f}% | Real: {daily_metrics['Realized PnL']:+8.0f} | "
                  f"Unreal: {daily_metrics['Unrealized PnL']:+8.0f} | Belief: {daily_metrics['Bayesian Belief']:.2%}")
            
            current_time += timedelta(days=1)
            # FORCE GC & Cache Clear to prevent RAM bloat
            if current_time.day % 7 == 0:
                torch.cuda.empty_cache()
                self.strategy.lab._feat_cache = {} 

        df = pd.DataFrame(results)
        df.to_csv("data/production_sim_results_hp.csv", index=False)
        logger.success(f"Simulation Finished. Final Account Value: ${nlv:,.2f}")
        return df

if __name__ == "__main__":
    sim = HighPerformanceProductionSim()
    sim.run(datetime(2023, 1, 1), datetime(2026, 5, 1))
