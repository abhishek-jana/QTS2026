import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from qts_core.logger import logger

class PortfolioGym(gym.Env):
    """
    UQTS-2026 Phase 3: The 'Chef' Environment.
    An RL environment for learning optimal portfolio policies.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, rankings_df, price_df, spy_df, initial_capital=100000.0):
        super(PortfolioGym, self).__init__()
        
        # CLEAN DATA & ENSURE MONOTONICITY
        self.rankings = rankings_df.sort_index().loc[~rankings_df.index.duplicated(keep='first')].fillna(0.0)
        self.prices = price_df.sort_index().loc[~price_df.index.duplicated(keep='first')].ffill().bfill().fillna(0.0)
        self.spy = spy_df.sort_index().loc[~spy_df.index.duplicated(keep='first')].ffill()
        self.initial_capital = initial_capital
        
        # Observation Space:
        # [Top 10 scores, Bottom 10 scores, Belief, Drawdown, SPY Vol, Current Leverage]
        self.observation_space = spaces.Box(
            low=np.array([-5.0]*24, dtype=np.float32), 
            high=np.array([5.0]*24, dtype=np.float32), 
            dtype=np.float32
        )
        
        # Action Space:
        # [Gross Leverage (0-2.5), Hedge Ratio (0-0.5), Concentration Index (0, 1, 2)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([2.5, 0.5, 2.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.dates = sorted(self.rankings.index.unique())
        self.ticker_list = [c for c in self.rankings.columns if c not in ['date']]
        self.prices_np = self.prices[self.ticker_list].values
        self.rankings_np = self.rankings[self.ticker_list].values
        
        self.current_step = 0
        self.reset_state()
        
    def reset_state(self):
        self.cash = self.initial_capital
        self.positions = {} # ticker_idx -> qty
        self.account_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_account_value = self.initial_capital
        self.hedge_qty = 0.0
        self.hedge_entry_p = 0.0

    def _get_obs(self):
        day_scores = self.rankings_np[self.current_step]
        sorted_scores = np.sort(day_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        drawdown = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
        
        lookback_idx = max(0, self.current_step - 21)
        spy_slice = self.spy.iloc[lookback_idx : self.current_step + 1]
        vol = spy_slice['close'].pct_change().std() if len(spy_slice) > 1 else 0.0
        
        belief = np.mean(top_10) - np.mean(bot_10)
        
        # SENIOR FIX: Sensor Alignment
        # current_lev represents GROSS EXPOSURE (Longs + Shorts) / NLV
        long_mv = self.account_value - self.cash
        short_mv = (self.hedge_qty * self.spy.iloc[self.current_step]['close']) if self.hedge_qty > 0 else 0
        current_lev = (abs(long_mv) + abs(short_mv)) / self.account_value if self.account_value > 0 else 0
        
        obs = np.concatenate([
            top_10, bot_10, [belief, drawdown, vol, current_lev]
        ]).astype(np.float32)
        
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reset_state()
        return self._get_obs(), {}

    def step(self, action):
        target_lev, hedge_ratio, concentration_idx = action
        n_stocks = [2, 5, 12][int(np.clip(concentration_idx, 0, 2))]
        
        # 1. Update Account Value based on Price Change from PREVIOUS step rebalance
        if self.current_step > 0:
            new_value = self.cash
            current_prices = self.prices_np[self.current_step]
            for t_idx, qty in self.positions.items():
                new_value += qty * current_prices[t_idx]
            
            if self.hedge_qty > 0:
                spy_p = self.spy.iloc[self.current_step]['close']
                new_value += self.hedge_qty * (self.hedge_entry_p - spy_p)

            self.account_value = new_value
            self.peak_value = max(self.peak_value, self.account_value)
            
        # 2. Execute Action (Rebalance for the NEXT step)
        day_rankings = self.rankings_np[self.current_step]
        top_k_indices = np.argpartition(day_rankings, -n_stocks)[-n_stocks:]
        
        target_notional = self.account_value * target_lev
        notional_per_stock = target_notional / n_stocks if n_stocks > 0 else 0
        
        self.cash = self.account_value
        self.positions = {}
        current_prices = self.prices_np[self.current_step]
        
        for t_idx in top_k_indices:
            p = current_prices[t_idx]
            if p > 0:
                qty = notional_per_stock / p
                self.positions[t_idx] = qty
                self.cash -= qty * p
        
        target_hedge_notional = self.account_value * hedge_ratio
        spy_price = self.spy.iloc[self.current_step]['close']
        self.hedge_qty = target_hedge_notional / spy_price if spy_price > 0 else 0
        self.hedge_entry_p = spy_price
        
        self.cash -= (self.account_value * (target_lev + hedge_ratio) * 0.0006) 
        
        # 3. V4.1 ASYMMETRIC REWARD: Chase Aggressive Alpha
        reward = 0
        done = False
        if self.current_step > 0:
            daily_ret = (self.account_value - self.prev_account_value) / self.prev_account_value if self.prev_account_value > 0 else 0
            
            spy_p0 = self.spy.iloc[self.current_step-1]['close']
            spy_p1 = self.spy.iloc[self.current_step]['close']
            spy_ret = (spy_p1 / spy_p0) - 1.0 if spy_p0 > 0 else 0
            
            alpha = daily_ret - spy_ret
            
            # Asymmetric Scaling: Reward winning 3x more than we punish trailing
            if alpha > 0:
                reward = alpha * 30.0 
            else:
                reward = alpha * 10.0 
            
            # Conviction Bonus for Leverage
            belief = np.mean(day_rankings[top_k_indices])
            if belief > 0.6 and target_lev > 1.8:
                reward += 0.05
            
            # Penalize excessive hedging in bull markets
            if spy_ret > 0.01 and hedge_ratio > 0.2:
                reward -= 0.02

            dd = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
            if dd < -0.12: reward -= 0.1 
            
            if self.account_value < self.initial_capital * 0.4: 
                reward -= 5.0
                done = True

        self.prev_account_value = self.account_value
        self.current_step += 1
        
        if self.current_step >= len(self.dates) - 1:
            done = True
            
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        print(f"Step: {self.current_step} | Value: ${self.account_value:,.2f} | Pos: {len(self.positions)}")
