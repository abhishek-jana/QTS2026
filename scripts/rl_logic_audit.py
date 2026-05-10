import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.simulation_engine import SimulationEngineV5
from qts_core.logger import logger

def verify_rl_logic():
    logger.info("🧪 Launching RL Logic Audit...")
    
    # 1. Initialize Simulation Engine (loads the new model)
    sim = SimulationEngineV5()
    if not sim.rl_pilot:
        logger.error("❌ New RL model not found!")
        return

    # 2. Run Out-of-Sample Test (2024-2026)
    # V4.2.1: STRICTLY MY MONEY (Max 1.0x Leverage)
    df = sim.run(datetime(2024, 1, 1), datetime(2026, 5, 1), max_leverage=1.0)
    
    if df is None or df.empty:
        logger.error("❌ Simulation failed.")
        return

    # 3. Audit: Concentration Versatility
    conc_dist = df['Conc'].value_counts(normalize=True)
    logger.info("--- 📊 Concentration Versatility Audit ---")
    for stocks, pct in conc_dist.items():
        logger.info(f"  {stocks} Stocks: {pct:.1%}")

    # 4. Audit: Leverage & Risk
    df['Drawdown'] = (df['NLV'] - df['NLV'].cummax()) / df['NLV'].cummax()
    max_dd = df['Drawdown'].min()
    max_gross = (df['Lev'] + df['Hedge']).max()
    
    logger.info("--- 🛡️ Leverage & Risk Audit ---")
    logger.info(f"  Max Gross Exposure: {max_gross:.2f}x (Limit: 1.00x)")
    logger.info(f"  Max Drawdown: {max_dd:.2%}")

    # 5. Final Verdict
    score = 0
    if len(conc_dist) > 1: score += 1 
    if max_gross <= 1.001: score += 1 # Allow for minor float precision
    if max_dd > -0.12: score += 1 # Unleveraged should have tighter DD control

    logger.info("--- 🏁 VERDICT ---")
    if score == 3:
        logger.success("✅ V4.2.1 VALIDATED: Strict 1.0x Cash Account behavior detected.")
    else:
        logger.warning(f"⚠️ V4.2.1 PARTIAL (Score {score}/3): Check leverage constraints.")

if __name__ == "__main__":
    verify_v4_2_logic()
