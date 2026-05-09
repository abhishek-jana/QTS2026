import os
import sys
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.rl_environment import PortfolioGym
from qts_core.logger import logger

def train_rl_pilot():
    parser = argparse.ArgumentParser(description="UQTS-2026 Phase 3 RL Trainer")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    args = parser.parse_args()

    logger.info("🎬 Initiating RL Pilot Training (Phase 3: The Chef)...")
    
    # 1. Load Pre-computed Data
    rankings_path = "data/rl/train_rankings.csv"
    prices_path = "data/rl/train_prices.csv"
    spy_path = "data/rl/train_spy.csv"
    
    if not os.path.exists(rankings_path):
        logger.error(f"❌ Missing {rankings_path}. Run scripts/precompute_rl_data.py first.")
        return
        
    rankings_df = pd.read_csv(rankings_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    
    # 2. Setup Environment
    env = DummyVecEnv([lambda: PortfolioGym(rankings_df, prices_df, spy_df)])
    
    # 3. Initialize Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, 
        device="auto"
    )
    
    # 4. Train
    total_timesteps = args.total_timesteps
    logger.info(f"🚀 Training for {total_timesteps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./models/rl_checkpoints/',
        name_prefix='rl_pilot'
    )
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 5. Save Final Policy
    os.makedirs("models", exist_ok=True)
    model.save("models/rl_pilot_final")
    logger.success("✅ RL Pilot Training Complete. Model saved to models/rl_pilot_final.zip")

if __name__ == "__main__":
    train_rl_pilot()
