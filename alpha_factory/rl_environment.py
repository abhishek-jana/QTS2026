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
        
        # CLEAN DATA
        self.rankings = rankings_df.fillna(0.0) 
        self.prices = price_df.ffill().bfill().fillna(0.0)
        self.spy = spy_df.ffill()
        self.initial_capital = initial_capital
        
        # Observation Space:
        # [Top 10 scores, Bottom 10 scores, Belief, Drawdown, SPY Vol, Current Leverage]
        # Total size: 10 + 10 + 1 + 1 + 1 + 1 = 24
        self.observation_space = spaces.Box(low=-5, high=5, shape=(24,), dtype=np.float32)
        
        # Action Space:
        # [Gross Leverage (0-2.5), Hedge Ratio (0-0.5), Concentration Index (0, 1, 2 for 2, 5, 12 stocks)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([2.5, 0.5, 2.0]), 
            dtype=np.float32
        )
        
        self.dates = sorted(self.rankings.index.unique())
        self.current_step = 0
        
        # State
        self.reset_state()
        
    def reset_state(self):
        self.cash = self.initial_capital
        self.positions = {} # ticker -> qty
        self.account_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_account_value = self.initial_capital

    def _get_obs(self):
        date = self.dates[self.current_step]
        day_scores = self.rankings.loc[date]
        
        # Top 10 and Bottom 10
        sorted_scores = day_scores.sort_values(ascending=False)
        top_10 = sorted_scores.head(10).values
        bot_10 = sorted_scores.tail(10).values
        
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))
        
        # Risk Metrics
        drawdown = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Robust SPY Vol proxy (last 21 steps)
        lookback_date = self.dates[max(0, self.current_step - 21)]
        spy_slice = self.spy[(self.spy.index >= lookback_date) & (self.spy.index <= date)]
        vol = spy_slice['close'].pct_change().std() if len(spy_slice) > 1 else 0.0
        
        belief = np.mean(top_10) - np.mean(bot_10)
        current_lev = (self.account_value - self.cash) / self.account_value if self.account_value > 0 else 0
        
        obs = np.concatenate([
            top_10, bot_10, [belief, drawdown, vol, current_lev]
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reset_state()
        return self._get_obs(), {}

    def step(self, action):
        target_lev, hedge_ratio, concentration_idx = action
        n_stocks = [2, 5, 12][int(np.clip(concentration_idx, 0, 2))]
        
        date = self.dates[self.current_step]
        
        # 1. Update Account Value based on Price Change from PREVIOUS step rebalance
        if self.current_step > 0:
            new_value = self.cash
            for t, qty in self.positions.items():
                p = self.prices.loc[date, t] if t in self.prices.columns else self.prices.iloc[self.current_step][t]
                new_value += qty * p
            
            self.account_value = new_value
            self.peak_value = max(self.peak_value, self.account_value)
            
        # 2. Execute Action (Rebalance for the NEXT step)
        day_scores = self.rankings.loc[date]
        sorted_tickers = day_scores.sort_values(ascending=False).index.tolist()
        top_picks = sorted_tickers[:n_stocks]
        
        target_notional = self.account_value * target_lev
        notional_per_stock = target_notional / n_stocks if n_stocks > 0 else 0
        
        # Close old, open new
        self.cash = self.account_value
        self.positions = {}
        for t in top_picks:
            if t in self.prices.columns:
                p = self.prices.loc[date, t]
                if p > 0:
                    qty = notional_per_stock / p
                    self.positions[t] = qty
                    self.cash -= qty * p
        
        # Borrowing cost / slippage proxy
        self.cash -= (self.account_value * target_lev * 0.0005) 
        
        # 3. Reward Calculation
        reward = 0
        done = False
        if self.current_step > 0:
            daily_ret = (self.account_value - self.prev_account_value) / self.prev_account_value if self.prev_account_value > 0 else 0
            dd = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
            
            reward = daily_ret
            if dd < -0.15: reward -= 0.1 # Heavy penalty for large DD
            
            if self.account_value < self.initial_capital * 0.4: # Bankruptcy
                reward -= 5.0
                done = True

        self.prev_account_value = self.account_value
        self.current_step += 1
        
        if self.current_step >= len(self.dates) - 1:
            done = True
            
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        print(f"Step: {self.current_step} | Value: ${self.account_value:,.2f} | Pos: {len(self.positions)}")
