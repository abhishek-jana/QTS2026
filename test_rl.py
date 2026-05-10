import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

rl_pilot = PPO.load("models/rl_pilot_final.zip", device="cpu")
# Create dummy observations
obs = np.zeros((1, 32), dtype=np.float32)
action, _ = rl_pilot.predict(obs, deterministic=True)
print("Action for zero obs:", action)

obs_rand = np.random.randn(1, 32).astype(np.float32)
action_rand, _ = rl_pilot.predict(obs_rand, deterministic=True)
print("Action for random obs:", action_rand)

