import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
from datetime import datetime

sys.path.append(os.getcwd())
from alpha_factory.rl_environment import PortfolioGym
from alpha_factory.simulation_engine import SimulationEngineV5

# Load data
rankings_df = pd.read_csv("data/rl/train_rankings.csv", index_col=0, parse_dates=True)
price_df = pd.read_csv("data/rl/train_prices.csv", index_col=0, parse_dates=True)
spy_df = pd.read_csv("data/rl/train_spy.csv", index_col=0, parse_dates=True)

env = PortfolioGym(rankings_df, price_df, spy_df)
obs_gym, _ = env.reset()

print("--- GYM OBS ---")
print(obs_gym)

# Move env to a specific date, e.g. 10th step
for i in range(10):
    obs_gym, _, _, _, _ = env.step([1.0, 0.0, 1.0])
print(f"--- GYM OBS STEP 10 ({env.dates[env.current_step]}) ---")
print(obs_gym)

