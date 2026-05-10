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

def perform_logic_audit(df):
    """Audits the agent logic based on simulation results."""
    logger.info("\n--- 🕵️ LOGIC AUDIT REPORT ---")
    
    if df is None or df.empty:
        logger.error("❌ No data available for audit.")
        return False

    # 1. Concentration Versatility
    conc_dist = df['Conc'].value_counts(normalize=True)
    logger.info("📊 Concentration Mix:")
    for stocks, pct in conc_dist.items():
        logger.info(f"  {stocks} Stocks: {pct:.1%}")

    # 2. Leverage & Risk
    df['Drawdown'] = (df['NLV'] - df['NLV'].cummax()) / df['NLV'].cummax()
    max_dd = df['Drawdown'].min()
    max_gross = (df['Lev'] + df['Hedge']).max()
    
    logger.info("🛡️ Risk & Leverage:")
    logger.info(f"  Max Gross Exposure: {max_gross:.2f}x (Constraint: 1.00x)")
    logger.info(f"  Max Drawdown: {max_dd:.2%}")

    # 3. Final Verdict
    score = 0
    if len(conc_dist) > 1: score += 1 
    if max_gross <= 1.001: score += 1 
    if max_dd > -0.12: score += 1 

    if score == 3:
        logger.success("✅ VERDICT: Institutional behavior VALIDATED.")
    else:
        logger.warning(f"⚠️ VERDICT: PARTIAL SUCCESS ({score}/3). Agent may be over-concentrated.")
    
    return score == 3

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
        
        # SENIOR FIX: Store RL metrics in JSON for later review
        import json
        summary = {
            "timestamp": datetime.now().isoformat(),
            "final_nlv": float(final_nlv),
            "total_return_pct": float(ret * 100),
            "baseline_return_pct": float(baseline_ret * 100),
            "spy_return_pct": float(spy_ret * 100),
            "alpha_pct": float(alpha * 100)
        }
        metrics_path = "data/strategy_metrics.json"
        existing_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f: existing_metrics = json.load(f)
            except Exception: pass
            
        existing_metrics["rl_evaluation"] = summary
        os.makedirs("data", exist_ok=True)
        with open(metrics_path, "w") as f: json.dump(existing_metrics, f, indent=4)
        logger.info(f"✨ RL Metrics stored to {metrics_path}")

        # Combined Run: Perform Audit on the SAME data
        perform_logic_audit(df)
    
    # --- STEP 2: MONTE CARLO STRESS TEST ---
    logger.info("\n--- STEP 2: MONTE CARLO STRESS TEST ---")
    mc = MonteCarloStressTest()
    steps = getattr(sim, 'last_steps', None)
    spy_df = getattr(sim, 'last_spy_df', None)
    mc.run_simulation(n_paths=10, backtest_mode=False, steps=steps, spy_df=spy_df)
    
    logger.info("\n" + "="*50)
    logger.success("✅ UNIFIED EVALUATION COMPLETE")
    logger.info("  Visual Reports Generated:")
    logger.info("  1. data/simulation_performance.png")
    logger.info("  2. data/monte_carlo_robustness.png")
    logger.info("="*50)

if __name__ == "__main__":
    run_rl_evaluation()
