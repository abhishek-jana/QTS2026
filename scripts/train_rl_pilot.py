import os
import sys
import argparse
import pandas as pd
import numpy as np
import multiprocessing
import random
import torch
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.rl_environment import PortfolioGym
from alpha_factory.meta_controller import BayesianMetaController
from qts_core.logger import logger

def make_env(rankings_df, prices_df, vols_df, spy_df, seed=42, rank=0):
    def _init():
        # SENIOR FIX: Set individual seeds for worker processes to ensure determinism
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        
        # SENIOR FIX (Priority 2): Instantiate real MetaController so the agent
        # "feels" model drift during training.
        mc = BayesianMetaController(prior_belief=0.75)
        
        env = PortfolioGym(rankings_df, prices_df, vols_df, spy_df, meta_controller=mc)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl_pilot(total_timesteps=2000000, seed=42, n_envs=None, resume=False):
    logger.info("🎬 Initiating HIGH-THROUGHPUT RL Training (Phase 4: The Survivor)...")
    
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

    # --- PERMANENT NAN SAFEGUARD ---
    rankings_df = rankings_df.ffill().bfill().fillna(0)
    prices_df = prices_df.ffill().bfill().fillna(0)
    vols_df = vols_df.ffill().bfill().fillna(0.02)
    spy_df = spy_df.ffill().bfill().fillna(0)
    
    if n_envs is None:
        n_envs = 12
    
    logger.info(f"🚀 Spawning {n_envs} parallel market environments with Seed {seed}...")
    
    env = SubprocVecEnv([make_env(rankings_df, prices_df, vols_df, spy_df, seed=seed, rank=i) for i in range(n_envs)])
    env = VecMonitor(env)
    
    # 3. Initialize Agent
    n_steps = 2048   
    # EFFICIENCY (Fix #12): Using 128 for finer policy updates during fine-tuning.
    batch_size = 128 
    
    current_steps = 0
    if resume:
        checkpoints = glob.glob("./models/rl_checkpoints/rl_pilot_v7_4_survivor_*.zip")
        if not checkpoints:
            logger.error("❌ Resume requested but no checkpoints found in ./models/rl_checkpoints/")
            return
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        logger.info(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, verbose=1, device="cpu")
        current_steps = model.num_timesteps
        logger.info(f"📈 Loaded model has already seen {current_steps:,} steps.")
    else:
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
            ent_coef=0.001,
            device="cpu",
            seed=seed
        )
    
    # 4. Train
    logger.info(f"📊 Training for {total_timesteps:,} additional steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // n_envs), 
        save_path='./models/rl_checkpoints/',
        name_prefix='rl_pilot_v7_4_survivor'
    )
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=not resume)
    
    # 5. Save Final Policy
    os.makedirs("models", exist_ok=True)
    model.save("models/rl_pilot_final")
    logger.success("✅ RL Pilot Training Complete. Model saved to models/rl_pilot_final.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UQTS-2026 Phase 4 Parallel RL Trainer")
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train_rl_pilot(total_timesteps=args.total_timesteps, seed=args.seed, n_envs=args.n_envs, resume=args.resume)

