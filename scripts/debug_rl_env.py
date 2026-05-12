import pandas as pd
import numpy as np
import os
import sys
import yaml
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.rl_environment import PortfolioGym
from qts_core.logger import logger

def debug_env():
    logger.info("🛠️ RL ENVIRONMENT HEALTH CHECK (10,000 Steps)")
    
    # 1. Load Data
    rankings_path = "data/rl/train_rankings.csv"
    prices_path = "data/rl/train_prices.csv"
    vols_path = "data/rl/train_stock_vols.csv"
    spy_path = "data/rl/train_spy.csv"
    
    rankings_df = pd.read_csv(rankings_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    vols_df = pd.read_csv(vols_path, index_col=0, parse_dates=True)
    spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    
    env = PortfolioGym(rankings_df, prices_df, vols_df, spy_df)
    obs, _ = env.reset()
    
    nan_found = False
    for i in range(10000):
        # Sample random action
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Check Observation
        if not np.all(np.isfinite(obs)):
            logger.error(f"STEP {i}: Observation contains NaN/Inf!")
            nan_found = True
            
        # Check Reward
        if not np.isfinite(reward):
            logger.error(f"STEP {i}: Reward is NaN/Inf!")
            nan_found = True
            
        # Check Internal State
        if not np.isfinite(env.account_value):
            logger.error(f"STEP {i}: Account Value is NaN!")
            nan_found = True
            
        if nan_found: break
        if done: env.reset()
            
    if not nan_found:
        logger.success("✅ Environment is clean for 10,000 random steps.")
    else:
        logger.error("❌ Environment failed the health check.")

if __name__ == "__main__":
    debug_env()
