import os
import sys
import argparse
import pandas as pd
import numpy as np
import multiprocessing
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.rl_environment_bare_metal import PortfolioGymBareMetal
from qts_core.logger import logger

def make_env(rankings_df, prices_df, vols_df, spy_df, seed=42, rank=0):
    def _init():
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        env = PortfolioGymBareMetal(rankings_df, prices_df, vols_df, spy_df)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl_pilot(total_timesteps=2000000, seed=42, n_envs=None):
    logger.info("🎬 Initiating EXPERIMENTAL Bare Metal Sandbox Training...")
    
    # 1. Load Pre-computed Data
    rankings_path = "data/rl/train_rankings.csv"
    prices_path = "data/rl/train_prices.csv"
    vols_path = "data/rl/train_stock_vols.csv"
    spy_path = "data/rl/train_spy.csv"
    
    if not os.path.exists(rankings_path):
        logger.error(f"❌ Missing {rankings_path}. Run 'python run.py rl data' first.")
        return
        
    rankings_df = pd.read_csv(rankings_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    vols_df = pd.read_csv(vols_path, index_col=0, parse_dates=True)
    spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)

    # --- SENIOR FIX (V7.4): PERMANENT NAN SAFEGUARD ---
    # Automatically clean incoming training data to prevent NaN-induced gradient collapse.
    logger.info("🛡️ Pre-Flight Check: Sanitizing RL Training Data...")
    rankings_df = rankings_df.ffill().bfill().fillna(0)
    prices_df = prices_df.ffill().bfill().fillna(0)
    vols_df = vols_df.ffill().bfill().fillna(0.02)
    spy_df = spy_df.ffill().bfill().fillna(0)
    
    # 2. Setup Parallel Environments (SENIOR OPTIMIZATION)
    # USER BENCHMARK: 12 logical threads proved faster than 6 physical cores 
    # for this specific memory/CPU configuration.
    if n_envs is None:
        n_envs = 12
    
    logger.info(f"🚀 Spawning {n_envs} parallel market environments with Seed {seed}...")
    
    env = SubprocVecEnv([make_env(rankings_df, prices_df, vols_df, spy_df, seed=seed, rank=i) for i in range(n_envs)])
    env = VecMonitor(env) # Track stats across all envs
    
    # 3. Initialize Agent
    n_steps = 2048   
    batch_size = 64 
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Policy crystallization
        device="cpu",
        seed=seed
    )
    
    # 4. Train
    logger.info(f"📊 Training for {total_timesteps} steps ({total_timesteps // (n_steps * n_envs)} iterations)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // n_envs), 
        save_path='./models/rl_checkpoints/',
        name_prefix='rl_pilot_bare_metal'
    )
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 5. Save Final Policy
    os.makedirs("models", exist_ok=True)
    model.save("models/rl_pilot_bare_metal")
    logger.success("✅ Bare Metal Training Complete. Model saved to models/rl_pilot_bare_metal.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UQTS-2026 Phase 4 Parallel RL Trainer")
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel envs (default: 6 physical cores)")
    args = parser.parse_args()
    train_rl_pilot(total_timesteps=args.total_timesteps, seed=args.seed, n_envs=args.n_envs)
