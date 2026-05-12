import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from alpha_factory.simulation_engine import SimulationEngineV5
from scripts.monte_carlo_regime_jitter import MonteCarloStressTest

def run_rl_evaluation():
    logger.info("🏁 Starting Unified RL Evaluation Pipeline (V5.1 Optimized)...")
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    
    # --- STEP 1: PERFORMANCE SIMULATION & LOGIC AUDIT ---
    logger.info("\n--- STEP 1: PERFORMANCE & LOGIC ---")
    sim = SimulationEngineV5()
    df = sim.run(start_date, end_date, max_leverage=1.0, backtest_mode=False)
    
    if df is not None and not df.empty:
        final_nlv = df['NLV'].iloc[-1]
        spy_final = df['SPY_NLV'].iloc[-1] if 'SPY_NLV' in df.columns else 100000.0
        ret = (final_nlv / 100000.0) - 1.0
        spy_ret = (spy_final / 100000.0) - 1.0
        alpha = ret - spy_ret
        
        # --- BASELINE COMPARISON ---
        logger.info("\n--- 🤖 RANKNET BASELINE COMPARISON ---")
        baseline_sim = SimulationEngineV5()
        baseline_sim.rl_pilot = None # Disable RL agent
        baseline_df = baseline_sim.run(start_date, end_date, max_leverage=1.0, backtest_mode=False)
        baseline_final = baseline_df['NLV'].iloc[-1] if baseline_df is not None else 100000.0
        baseline_ret = (baseline_final / 100000.0) - 1.0
        
        logger.info(f"  Simulation Range: {start_date.date()} -> {df['Date'].iloc[-1].date()}")
        logger.info(f"  💰 RL AGENT FINAL CAPITAL: ${final_nlv:,.2f} ({ret:.2%})")
        logger.info(f"  🤖 RANKNET BASELINE:       ${baseline_final:,.2f} ({baseline_ret:.2%})")
        logger.info(f"  📊 SPY BENCHMARK:          ${spy_final:,.2f} ({spy_ret:.2%})")
        logger.info(f"  🎯 RL AGENT TOTAL ALPHA:   {alpha:.2%}")
        
        # --- EXECUTION AUDIT (The requested block) ---
        avg_exp_rl = df['Lev'].mean()
        avg_exp_base = baseline_df['Lev'].mean() if baseline_df is not None else 1.0
        rl_churn = (df['Lev'].diff().abs() > 0.05).sum()
        base_churn = (baseline_df['Lev'].diff().abs() > 0.05).sum() if baseline_df is not None else 0
        
        # Calculate Max Drawdown
        def get_mdd(series):
            return (series / series.cummax() - 1).min()
            
        rl_mdd = get_mdd(df['NLV'])
        base_mdd = get_mdd(baseline_df['NLV']) if baseline_df is not None else 0.0

        logger.info("\n" + "="*50)
        logger.info("🛡️ EXECUTION PERFORMANCE AUDIT:")
        logger.info(f"  RL Average Net Exposure: {avg_exp_rl:.2f}x")
        logger.info(f"  Baseline Net Exposure:   {avg_exp_base:.2f}x")
        logger.info(f"  RL Churn Events:         {rl_churn} days")
        logger.info(f"  Baseline Churn Events:   {base_churn} days")
        logger.info(f"  RL Max Drawdown:         {rl_mdd:.2%}")
        logger.info(f"  Baseline Max Drawdown:   {base_mdd:.2%}")
        logger.info("="*50 + "\n")
        
        # Store metrics
        import json
        summary = {
            "timestamp": datetime.now().isoformat(),
            "final_nlv": float(final_nlv),
            "total_return_pct": float(ret * 100),
            "baseline_return_pct": float(baseline_ret * 100),
            "spy_return_pct": float(spy_ret * 100),
            "alpha_pct": float(alpha * 100),
            "rl_avg_exposure": float(avg_exp_rl),
            "rl_mdd": float(rl_mdd)
        }
        os.makedirs("data", exist_ok=True)
        with open("data/strategy_metrics.json", "w") as f:
            json.dump({"rl_evaluation": summary}, f, indent=2)
            
        # --- STEP 2: MONTE CARLO STRESS TEST ---
        logger.info("\n--- STEP 2: MONTE CARLO STRESS TEST ---")
        mc = MonteCarloStressTest()
        steps = sim.last_steps
        spy_df = sim.last_spy_df
        mc.run_simulation(n_paths=10, backtest_mode=False, steps=steps, spy_df=spy_df)

        # --- STEP 3: UNIFIED PERFORMANCE PLOT ---
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7), facecolor='#050505')
        ax = plt.gca(); ax.set_facecolor('#050505')
        
        plt.plot(df['Date'], df['NLV'], color='#10b981', lw=2.5, label='RL Survivor (V7.4)')
        if baseline_df is not None:
            plt.plot(baseline_df['Date'], baseline_df['NLV'], color='#3b82f6', lw=1.5, alpha=0.8, label='RankNet Baseline')
        
        plt.plot(df['Date'], df['SPY_NLV'], color='#475569', ls='--', lw=1.5, label='SPY Benchmark')
        
        plt.title("Master Sniper V7.4: Survivor Alpha vs Benchmark", color='white', fontsize=16, fontweight='bold')
        plt.xlabel("Date", color='white'); plt.ylabel("Capital ($)", color='white')
        plt.grid(True, alpha=0.1, color='white')
        plt.legend(facecolor='#050505', edgecolor='white', labelcolor='white')
        plt.tick_params(colors='white')
        
        os.makedirs("data", exist_ok=True)
        plt.savefig("data/simulation_performance.png")
        plt.close()
        logger.info("🎨 Unified Performance Plot saved to data/simulation_performance.png")

    logger.info("\n" + "="*50)
    logger.success("✅ UNIFIED EVALUATION COMPLETE")
    logger.info("  Visual Reports Generated:")
    logger.info("  1. data/simulation_performance.png")
    logger.info("  2. data/monte_carlo_robustness.png")
    logger.info("="*50)

if __name__ == "__main__":
    run_rl_evaluation()
