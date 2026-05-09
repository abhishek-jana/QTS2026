import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger

def run_turbo_sim():
    """
    ULTRA-FAST MACRO SIMULATOR.
    Uses pre-computed IC and Stock Returns to benchmark Logic V3/V4.
    """
    logger.info("🚀 STARTING TURBO SIM (MACRO MODE)...")
    
    # 1. Load Data
    # We use the backtest_results which contain the ICs
    if not os.path.exists("data/backtest_results.csv"):
        logger.error("❌ Need data/backtest_results.csv. Run 'python run.py --eval-only' first.")
        return
        
    df = pd.read_csv("data/backtest_results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Load SPY for benchmark
    # We'll pull SPY from the DB once for the benchmark
    import duckdb
    conn = duckdb.connect("data/uqts_v2_intraday.ddb", read_only=True)
    spy_df = conn.execute("SELECT event_time, close FROM market_data WHERE ticker = 'SPY' ORDER BY event_time").df()
    spy_df['event_time'] = pd.to_datetime(spy_df['event_time'])
    conn.close()

    # 2. Setup Scenarios
    scenarios = [
        {"name": "V3: High-Octane (2.0x)", "lev": 2.0, "threshold": 0.15, "ic_mult": 0.35},
        {"name": "V2: Unleashed (1.2x)", "lev": 1.2, "threshold": 0.30, "ic_mult": 0.25},
        {"name": "SPY: Benchmark", "lev": 1.0, "threshold": 0.0, "ic_mult": 0.0}
    ]

    results = {s['name']: [100000.0] for s in scenarios}
    dates = [df['date'].iloc[0]]
    
    # Start Portfolio Values
    p_vals = {s['name']: 100000.0 for s in scenarios}
    belief = 0.5
    
    # 3. Vectorized Simulation Loop
    for i in range(len(df)-1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i+1]
        dt = next_row['date']
        
        # Bayesian Update (Simplified)
        ic = curr_row['challenger_ic']
        l_v = 1 / (1 + np.exp(-5 * (ic - 0.01)))
        belief = (l_v * belief) / ((l_v * belief) + ((1-l_v) * (1-belief)))
        belief = max(0.05, min(0.95, belief))
        
        # SPY Period Return
        spy_start = spy_df[spy_df['event_time'] <= curr_row['date']]['close'].iloc[-1]
        spy_end = spy_df[spy_df['event_time'] <= next_row['date']]['close'].iloc[-1]
        spy_ret = (spy_end / spy_start) - 1.0
        
        for s in scenarios:
            if "SPY" in s['name']:
                p_vals[s['name']] *= (1 + spy_ret)
            else:
                # Alpha Proxy: IC * Volatility + SPY Beta
                alpha = ic * s['ic_mult'] 
                active_lev = s['lev'] if belief > s['threshold'] else 0.0
                period_ret = (1.0 * spy_ret) + alpha
                p_vals[s['name']] *= (1 + period_ret * active_lev)
            
            results[s['name']].append(p_vals[s['name']])
        
        dates.append(dt)

    # 4. Final Comparison & Plotting
    res_df = pd.DataFrame(results, index=dates)
    
    plt.figure(figsize=(14, 8))
    plt.plot(res_df.index, res_df['V3: High-Octane (2.0x)'], label='High-Octane (2.0x)', color='#2ecc71', lw=2.5)
    plt.plot(res_df.index, res_df['V2: Unleashed (1.2x)'], label='Unleashed (1.2x)', color='#3498db', lw=2)
    plt.plot(res_df.index, res_df['SPY: Benchmark'], label='S&P 500', color='#bdc3c7', linestyle='--')
    
    plt.title("UQTS-2026: Turbo Sim Analysis (2023-2026)", fontsize=16, fontweight='bold')
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig("data/turbo_sim_results.png")
    
    logger.success(f"Turbo Sim Complete! Results plotted to data/turbo_sim_results.png")
    print("\n" + "="*40)
    print("FINAL BALANCES (2026)")
    print("="*40)
    for col in res_df.columns:
        print(f"{col:25}: ${res_df[col].iloc[-1]:,.2f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_turbo_sim()
