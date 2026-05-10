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

from alpha_factory.rl_environment import PortfolioGym
from qts_core.logger import logger

def make_env(rankings_df, prices_df, spy_df, seed=42, rank=0):
    def _init():
        # SENIOR FIX: Set individual seeds for worker processes to ensure determinism
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        env = PortfolioGym(rankings_df, prices_df, spy_df)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl_pilot(total_timesteps=100000, seed=42):
    logger.info("🎬 Initiating HIGH-THROUGHPUT RL Training (Phase 3: The Chef)...")
    
    # 1. Load Pre-computed Data
    rankings_path = "data/rl/train_rankings.csv"
    prices_path = "data/rl/train_prices.csv"
    spy_path = "data/rl/train_spy.csv"
    
    if not os.path.exists(rankings_path):
        logger.error(f"❌ Missing {rankings_path}. Run 'python run.py --precompute-rl' first.")
        return
        
    rankings_df = pd.read_csv(rankings_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    
    # 2. Setup Parallel Environments (SENIOR OPTIMIZATION)
    n_envs = min(multiprocessing.cpu_count(), 12)
    logger.info(f"🚀 Spawning {n_envs} parallel market environments with Seed {seed}...")
    
    env = SubprocVecEnv([make_env(rankings_df, prices_df, spy_df, seed=seed, rank=i) for i in range(n_envs)])
    env = VecMonitor(env) # Track stats across all envs
    
    # 3. Initialize Agent
    # SENIOR FIX: Force CPU for MLP (lower latency than GPU for small vectors)
    n_steps = 1024   # Shorter rollouts per env for more frequent updates
    batch_size = 256 # Larger batch size for better CPU utilization
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, 
        device="cpu",
        seed=seed # SENIOR FIX: Lock PPO policy weights and sampling
    )
    
    # 4. Train
    logger.info(f"📊 Training for {total_timesteps} steps ({total_timesteps // (n_steps * n_envs)} iterations)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 10000 // n_envs), 
        save_path='./models/rl_checkpoints/',
        name_prefix='rl_pilot_v4'
    )
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 5. Save Final Policy
    os.makedirs("models", exist_ok=True)
    model.save("models/rl_pilot_final")
    logger.success("✅ RL Pilot Training Complete. Model saved to models/rl_pilot_final.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UQTS-2026 Phase 3 Parallel RL Trainer")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_rl_pilot(total_timesteps=args.total_timesteps, seed=args.seed)
