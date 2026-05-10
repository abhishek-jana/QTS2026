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

def audit_rl():
    logger.info("🕵️ Starting RL Agent Audit...")
    
    sim = SimulationEngineV5()
    if not sim.rl_pilot:
        logger.error("❌ RL Pilot not found! Audit aborted.")
        return

    # Run for a representative period
    df = sim.run(datetime(2024, 1, 1), datetime(2024, 6, 1), max_leverage=2.5)
    
    if df is None or df.empty:
        logger.error("❌ Simulation returned no data.")
        return

    conc_counts = df['Conc'].value_counts()
    logger.info("--- Concentration Distribution ---")
    for val, count in conc_counts.items():
        logger.info(f"Stocks: {val} | Count: {count} ({count/len(df):.1%})")

    max_lev = df['Lev'].max()
    logger.info(f"--- Leverage Audit ---")
    logger.info(f"Max Leverage observed: {max_lev:.2f}x")
    
    if max_lev <= 2.0:
        logger.success(f"✅ Agent respects the 2.0x (200%) 'soft' limit perfectly (Target 2.5x).")
    else:
        logger.warning(f"⚠️ Agent uses aggressive leverage: {max_lev:.2f}x")

    if 12 in conc_counts and conc_counts[12] / len(df) > 0.8:
        logger.warning("🚨 'Lazy Agent' Alert: Agent picks 12 stocks >80% of the time. It is playing too safe.")
    elif 2 in conc_counts and conc_counts[2] / len(df) > 0.5:
        logger.info("🎯 'Hunter' Profile: Agent frequently picks high concentration (2 stocks).")
    else:
        logger.info("⚖️ 'Balanced' Profile: Agent varies its concentration.")

if __name__ == "__main__":
    audit_rl()
