import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import yaml
from datetime import datetime, timedelta
from qts_core.logger import logger
import os

class PortfolioGym(gym.Env):
    """
    Expert-Grade Reinforcement Learning Environment for V7.4 Sniper.
    Features:
    - Smart Binary Risk Toggle (100% On / 100% Cash)
    - Shield & Sword Reward Logic (Reward defensive alpha + offensive alpha)
    - Volatility Velocity & SPY Trend Sensors
    - Institutional Friction & Turnover Constraints
    """
    def __init__(self, rankings_df, prices_df, vols_df, spy_df, config_path="config.yaml"):
        super(PortfolioGym, self).__init__()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.rankings = rankings_df
        self.prices = prices_df
        self.vols = vols_df
        self.spy = spy_df
        
        self.initial_capital = 100000.0
        
        # Action Space: [Risk Toggle, Concentration Index, Execution Trigger]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 3.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Observation Space: 32 macro/micro sensors
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(32,), dtype=np.float32)
        
        self.dates = sorted(self.rankings.index.unique())
        self.ticker_list = [c for c in self.rankings.columns if c not in ['date']]
        self.prices_np = self.prices.values
        self.rankings_np = self.rankings.values
        self.vols_np = self.vols.values
        
        # Pre-calculate Volatility Velocity
        self.spy_vol_velocity = self.spy['vol_21'].diff().rolling(5).mean().fillna(0).values
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.account_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {} # {ticker_idx: qty}
        self.account_history = [self.initial_capital]
        self.spy_history = [self.spy.iloc[0]['close']]
        self.peak_value = self.initial_capital
        self.hedge_qty = 0.0
        self.hedge_entry_p = 0.0
        self.last_target_lev = 1.0
        self.last_n_stocks = 12
        return self._get_obs(), {}

    def _get_obs(self):
        # Ensure we don't exceed bounds
        step = min(self.current_step, len(self.dates) - 1)
        
        scores = self.rankings_np[step]
        scaled_scores = scores * 100.0
        sorted_scores = np.sort(scaled_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        safe_nlv = max(self.account_value, 1.0)
        drawdown = (self.account_value - self.peak_value) / (self.peak_value + 1e-6)
        spy_row = self.spy.iloc[step]
        
        current_prices = self.prices_np[step]
        pos_mv = sum([qty * current_prices[t_idx] for t_idx, qty in self.positions.items()])
        current_lev = pos_mv / safe_nlv
        
        belief = np.mean(top_10)
        vol_vel = self.spy_vol_velocity[step] * 1000.0
        spy_trend = (spy_row.get('ma_ratio', 1.0) - 1.0) * 10.0
        
        dow = self.dates[step].weekday() / 6.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, spy_row.get('vol_21', 0.02), current_lev],
            [vol_vel, spy_trend, (spy_row.get('rsi_14', 50.0)-50)/50, spy_row.get('ret', 0.0)],
            [self.cash/safe_nlv, 0.0, 1.0, dow]
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        risk_toggle, concentration_idx, exec_trigger = action
        if not np.all(np.isfinite(action)):
            risk_toggle, concentration_idx, exec_trigger = 1.0, 1.0, 0.0

        current_dt = self.dates[self.current_step]
        should_rebalance = (exec_trigger > 0.7) or (current_dt.weekday() == 0)
        
        if should_rebalance:
            self.last_target_lev = 1.0 if risk_toggle > 0.5 else 0.0
            self.last_n_stocks = [5, 8, 12, 15][int(np.clip(concentration_idx, 0, 3.99))]

        current_prices = self.prices_np[self.current_step]
        spy_price = self.spy.iloc[self.current_step]['close']
        pos_mv = sum([qty * current_prices[t_idx] for t_idx, qty in self.positions.items()])
        self.account_value = self.cash + pos_mv + self.hedge_qty * (self.hedge_entry_p - spy_price)
        self.peak_value = max(self.peak_value, self.account_value)

        turnover_notional = 0.0
        if should_rebalance:
            scores = self.rankings_np[self.current_step]
            top_k_indices = np.argsort(scores)[-self.last_n_stocks:][::-1]
            top_scores = scores[top_k_indices]
            exp_scores = np.exp((top_scores - np.max(top_scores)) / 0.5)
            weights = exp_scores / (np.sum(exp_scores) + 1e-9)
            
            target_notion = self.account_value * self.last_target_lev
            new_positions = {}
            for t_idx, old_qty in self.positions.items():
                if t_idx not in top_k_indices: turnover_notional += old_qty * current_prices[t_idx]
            
            for i, t_idx in enumerate(top_k_indices):
                p = max(current_prices[t_idx], 1e-6)
                t_qty = (target_notion * weights[i]) / p
                c_qty = self.positions.get(t_idx, 0.0)
                if c_qty == 0 or abs(t_qty - c_qty) / (c_qty + 1e-6) > 0.15:
                    turnover_notional += abs(t_qty - c_qty) * p
                    new_positions[t_idx] = t_qty
                else:
                    new_positions[t_idx] = c_qty
            self.positions = new_positions
            self.account_value -= turnover_notional * 0.0005

        pos_mv_now = sum([q * current_prices[t] for t, q in self.positions.items()])
        self.cash = self.account_value - pos_mv_now
        self.account_history.append(self.account_value)
        self.spy_history.append(spy_price)

        # SHIELD & SWORD REWARD
        reward = 0.0
        if self.current_step > 5:
            prev_acc = self.account_history[self.current_step-1]
            prev_spy = self.spy_history[self.current_step-1]
            agent_ret_1d = (self.account_value / (prev_acc + 1e-6)) - 1.0
            spy_ret_1d = (spy_price / (prev_spy + 1e-6)) - 1.0
            
            current_lev = pos_mv_now / (self.account_value + 1e-6)
            vol_vel = self.spy_vol_velocity[self.current_step]
            
            # --- PHASE 4.5: ABSOLUTE ALPHA (Unconditional) ---
            # Reward beating the SPY index whether we are long or in cash.
            # If SPY is -2% and Agent is 0% (Cash), Alpha is +2%. Reward = +20.0
            # This makes de-risking the most profitable decision during crashes.
            alpha_1d = agent_ret_1d - spy_ret_1d
            reward += float(alpha_1d) * 1000.0
            
            # --- THE SMART SIGNAL GATE (Survivor V3) ---
            top_conviction = np.mean(np.sort(self.rankings_np[self.current_step])[-5:])
            is_signal_failing = (vol_vel > 0.0003) and (top_conviction < 0.008)
            
            # --- NEW: ABSOLUTE BLEED PENALTY ---
            # Regardless of the index, losing real money is a sin.
            if agent_ret_1d < -0.01:
                reward -= abs(agent_ret_1d) * 500.0
            
            # --- SMART FEE-FEAR PENALTY ---
            if is_signal_failing:
                if exec_trigger > 0.5: reward -= 2.0 
                if current_lev > 0.1:  reward -= (current_lev * 5.0) 
            
            # 3. PUNISH DUMB CASH: Only if market is healthy and conviction is high
            if current_lev < 0.10 and top_conviction > 0.015 and not is_signal_failing and spy_ret_1d > 0:
                reward -= 0.5
                
            # Standard Friction
            reward -= (turnover_notional / (self.account_value + 1e-6)) * 2.5
            
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1 or self.account_value < self.initial_capital * 0.3
        reward = np.clip(np.nan_to_num(reward, nan=-1.0), -20.0, 20.0)
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        print(f"Step: {self.current_step} | Value: ${self.account_value:,.2f} | Pos: {len(self.positions)}")
