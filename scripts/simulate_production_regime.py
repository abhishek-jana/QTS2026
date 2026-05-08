import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

class HeadlessProductionSim:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'])
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        # State
        self.starting_capital = 100000.0
        self.sim_cash = 100000.0
        self.sim_positions = {} # Ticker -> Qty
        self.sim_avg_costs = {}
        self.sim_realized_pnl = 0.0
        self.order_log = []
        self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
        
    def _get_latest_price(self, ticker, current_time):
        try:
            view = self.data_engine.get_pit_view(ticker, current_time)
            if not view.empty: return float(view['close'].iloc[-1])
            lookback = current_time - timedelta(days=1)
            view = self.data_engine.get_pit_view(ticker, lookback)
            if not view.empty: return float(view['close'].iloc[-1])
        except: pass
        return 0.0

    def _get_market_regime(self, current_time):
        try:
            p_now_view = self.data_engine.get_pit_view('SPY', current_time)
            p_past_view = self.data_engine.get_pit_view('SPY', current_time - timedelta(days=21))
            if not p_now_view.empty and not p_past_view.empty:
                p_now = float(p_now_view['close'].iloc[-1])
                p_past = float(p_past_view['close'].iloc[-1])
                return "BULL" if p_now > p_past else "BEAR"
        except: pass
        return "BULL"

    def run(self, start_date, end_date):
        logger.info(f"🚀 STARTING HEADLESS PRO SIM: {start_date.date()} -> {end_date.date()}")
        
        current_time = start_date
        results = []
        
        while current_time <= end_date:
            # 1. Get House View & Belief
            view = self.strategy.get_current_rankings(as_of=current_time)
            if view['status'] != "OK":
                current_time += timedelta(days=1)
                continue
                
            belief = float(view['belief_score'])
            belief = max(0.05, min(0.95, belief))
            
            # 2. Calculate NLV
            unrealized_pnl = 0.0
            gross_notional = 0.0
            net_notional = 0.0
            for t, qty in self.sim_positions.items():
                p = self._get_latest_price(t, current_time) or self.sim_avg_costs[t]
                entry = self.sim_avg_costs[t]
                pnl = (p - entry) * qty if qty > 0 else (entry - p) * abs(qty)
                unrealized_pnl += pnl
                pos_val = qty * p
                gross_notional += abs(pos_val)
                net_notional += pos_val
                
            nlv = self.starting_capital + self.sim_realized_pnl + unrealized_pnl
            
            # 3. Execution Logic (Regime-Leveraged + Trimming)
            threshold = 0.65
            total_tickers = len(view['ladder'])
            entry_size = max(1, total_tickers // 10)
            hold_size = max(1, int(total_tickers * 0.15))
            
            regime = self._get_market_regime(current_time)
            allow_shorts = (regime == "BEAR")
            
            top_entry = [e['ticker'] for e in view['ladder'][:entry_size]]
            bottom_entry = [e['ticker'] for e in view['ladder'][-entry_size:]] if allow_shorts else []
            top_hold = [e['ticker'] for e in view['ladder'][:hold_size]]
            bottom_hold = [e['ticker'] for e in view['ladder'][-hold_size:]]

            # Sizing
            leverage = min(1.5, belief * 1.2)
            target_total_notional = nlv * leverage
            notional_per_slot = target_total_notional / 10
            
            # A. Exit/Hysteresis Logic
            for t in list(self.sim_positions.keys()):
                qty = self.sim_positions[t]
                should_close = False
                if qty > 0 and t not in top_hold: should_close = True
                if qty < 0 and (t not in bottom_hold or not allow_shorts): should_close = True
                
                if should_close:
                    p = self._get_latest_price(t, current_time)
                    if p > 0:
                        exec_qty = abs(qty)
                        impact = exec_qty * p
                        # Fill immediately for headless sim
                        pnl = (p - self.sim_avg_costs[t]) * exec_qty if qty > 0 else (self.sim_avg_costs[t] - p) * exec_qty
                        self.sim_realized_pnl += pnl
                        if qty > 0: self.sim_cash += impact
                        else: self.sim_cash -= impact
                        del self.sim_positions[t]
                        del self.sim_avg_costs[t]
                        self.oms_queue["filled"] += 1

            # B. Trimming & New Entry
            if belief > threshold:
                trade_intents = [(t, "BUY") for t in top_entry] + [(t, "SHORT") for t in bottom_entry]
                for ticker, intent in trade_intents:
                    p = self._get_latest_price(ticker, current_time)
                    if p <= 0: continue
                    
                    if ticker in self.sim_positions:
                        # Trimming
                        qty = self.sim_positions[ticker]
                        current_notional = abs(qty * p)
                        if current_notional > (notional_per_slot * 1.25):
                            trim_qty = int((current_notional - notional_per_slot) / p)
                            if trim_qty > 0:
                                pnl = (p - self.sim_avg_costs[ticker]) * trim_qty if qty > 0 else (self.sim_avg_costs[ticker] - p) * trim_qty
                                self.sim_realized_pnl += pnl
                                if qty > 0: self.sim_cash += (trim_qty * p)
                                else: self.sim_cash -= (trim_qty * p)
                                self.sim_positions[ticker] -= (trim_qty if qty > 0 else -trim_qty)
                                self.oms_queue["filled"] += 1
                    else:
                        # New Entry
                        target_notional = notional_per_slot
                        if intent == "BUY":
                            target_notional = min(target_notional, self.sim_cash * 0.9)
                        
                        qty = int(target_notional / p)
                        if qty > 0:
                            impact = qty * p
                            if intent == "BUY": self.sim_cash -= impact
                            else: self.sim_cash += impact
                            self.sim_positions[ticker] = qty if intent == "BUY" else -qty
                            self.sim_avg_costs[ticker] = p
                            self.oms_queue["filled"] += 1

            # Record Daily Stats
            results.append({
                "Date": current_time.strftime("%Y-%m-%d"),
                "Active Pos": len(self.sim_positions),
                "Gross Exp (%)": (gross_notional / nlv * 100) if nlv > 0 else 0,
                "Account Value": nlv,
                "Buying Power": self.sim_cash,
                "ROE (Invested)": (self.sim_realized_pnl + unrealized_pnl) / (gross_notional if gross_notional > 0 else 1) * 100,
                "Realized PnL": self.sim_realized_pnl,
                "Unrealized PnL": unrealized_pnl,
                "Bayesian Belief": belief,
                "OMS Filled": self.oms_queue["filled"]
            })
            
            if current_time.day == 1:
                logger.info(f"Simulated through {current_time.date()} | NLV: ${nlv:,.2f}")
                
            current_time += timedelta(days=1)
            
        df = pd.DataFrame(results)
        df.to_csv("data/production_sim_results_2026.csv", index=False)
        logger.success("Headless simulation complete. Results saved to data/production_sim_results_2026.csv")
        return df

if __name__ == "__main__":
    sim = HeadlessProductionSim()
    sim.run(datetime(2023, 1, 1), datetime(2026, 5, 1))
