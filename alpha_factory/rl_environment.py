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
    UQTS-2026 Phase 3: The 'Chef' Environment.
    V5.3: "Master Chief" (Level 3 Risk Parity Edition)
    - 32 Sensors (No Look-Ahead Leaks)
    - 1.0x Max Gross Exposure (Cash Account)
    - Minimum 5-stock diversification
    - Turnover-Based Friction
    - Rolling 5-day Alpha Reward
    - Level 3 Risk Parity Sizing (Backward Compatible)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, rankings_df, price_df, spy_df, initial_capital=100000.0):
        super(PortfolioGym, self).__init__()
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.risk_parity = config.get('execution_muscle', {}).get('risk_parity_sizing', True)
        
        self.rankings = rankings_df.sort_index().loc[~rankings_df.index.duplicated(keep='first')].fillna(0.0)
        self.prices = price_df.sort_index().loc[~price_df.index.duplicated(keep='first')].ffill().bfill().fillna(0.0)
        self.spy = spy_df.sort_index().loc[~spy_df.index.duplicated(keep='first')].ffill().fillna(0.0)
        
        # Load Pre-computed Volatilities for Level 3
        vol_path = "data/rl/train_stock_vols.csv"
        if os.path.exists(vol_path):
            vols_df = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            self.vols = vols_df.sort_index().loc[~vols_df.index.duplicated(keep='first')].ffill().bfill().fillna(0.02)
        else:
            logger.warning("train_stock_vols.csv not found. Falling back to constant volatility.")
            self.vols = pd.DataFrame(0.02, index=self.prices.index, columns=self.prices.columns)
            
        self.initial_capital = initial_capital
        
        # Observation Space: 32 Sensors
        self.observation_space = spaces.Box(
            low=np.array([-10.0]*32, dtype=np.float32), 
            high=np.array([10.0]*32, dtype=np.float32), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 3.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.dates = sorted(self.rankings.index.unique())
        self.ticker_list = [c for c in self.rankings.columns if c not in ['date']]
        self.prices_np = self.prices[self.ticker_list].values
        self.rankings_np = self.rankings[self.ticker_list].values
        self.vols_np = self.vols[self.ticker_list].values
        
        self.current_step = 0
        self.reset_state()
        
    def reset_state(self):
        self.cash = self.initial_capital
        self.positions = {} 
        self.account_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_account_value = self.initial_capital
        self.hedge_qty = 0.0
        self.hedge_entry_p = 0.0
        self.portfolio_returns = [0.0] * 10
        self.account_history = [self.initial_capital] * 10
        self.spy_history = [float(self.spy.iloc[0]['close'])] * 10

    def _get_obs(self):
        day_scores = self.rankings_np[self.current_step]
        sorted_scores = np.sort(day_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        # V5.2 GHOST-PROTOCOL: Use T-1 (Yesterday) for ALL macro sensors
        current_date = self.dates[self.current_step]
        spy_mask_yesterday = self.spy.index < current_date
        spy_prev = self.spy[spy_mask_yesterday].iloc[-1] if spy_mask_yesterday.any() else self.spy.iloc[0]
        
        # Current SPY for MtM
        spy_mask_current = self.spy.index <= current_date
        spy_curr = self.spy[spy_mask_current].iloc[-1] if spy_mask_current.any() else self.spy.iloc[0]
        
        # 1. BASE SENSORS (Yesterday's State)
        belief = np.mean(top_10) - np.mean(bot_10)
        drawdown = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
        vol = spy_prev.get('vol_21', 0.0)
        
        long_mv = self.account_value - self.cash
        short_mv = (self.hedge_qty * spy_curr['close']) if self.hedge_qty > 0 else 0
        current_lev = (abs(long_mv) + abs(short_mv)) / self.account_value if self.account_value > 0 else 0
        
        # 2. MACRO SENSORS (Yesterday's Data)
        vov = spy_prev.get('vov_21', 0.0) * 100.0
        ma_ratio = (spy_prev.get('ma_ratio', 1.0) - 1.0) * 10.0
        rsi = (spy_prev.get('rsi_14', 50.0) - 50.0) / 50.0
        spy_ret_yest = spy_prev.get('ret', 0.0) * 10.0
        
        # 3. EXECUTION SENSORS
        cash_ratio = self.cash / self.account_value if self.account_value > 0 else 1.0
        
        # Alpha Decay
        prev_wk_idx = max(0, self.current_step - 5)
        prev_top_mean = np.mean(np.sort(self.rankings_np[prev_wk_idx])[-10:])
        alpha_decay = (np.mean(top_10) - prev_top_mean)
        
        # Relative Strength (RS) - Yesterday's Relative Performance
        recent_p_rets = self.portfolio_returns[-5:]
        p_ret_avg = np.mean(recent_p_rets)
        rs = p_ret_avg / (spy_prev.get('ret', 0.001) + 1e-6)
        
        dow = current_date.weekday() / 6.0
        
        obs = np.concatenate([
            top_10, bot_10, 
            [belief, drawdown, vol, current_lev], 
            [vov, ma_ratio, rsi, spy_ret_yest],      
            [cash_ratio, alpha_decay, rs, dow]     
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reset_state()
        return self._get_obs(), {}

    def step(self, action):
        target_lev, hedge_ratio, concentration_idx = action
        
        # --- SENIOR FIX (V5.8): GROWTH HUNTER (LONG-ONLY) ---
        # We force hedge_ratio to 0 to test if the RankNet can actually pick winners.
        hedge_ratio = 0.0
        
        # Enforce 1.0x Gross Limit (Simplified for Long-Only)
        target_lev = np.clip(target_lev, 0.0, 1.0)
        
        conc_idx = int(np.clip(concentration_idx, 0, 2.999))
        n_stocks = [5, 8, 12][conc_idx]
        
        # 1. Update NLV
        if self.current_step > 0:
            current_prices = self.prices_np[self.current_step]
            pos_mv = sum([qty * current_prices[t_idx] for t_idx, qty in self.positions.items()])
            spy_p = self.spy.iloc[self.current_step]['close']
            hedge_pnl = self.hedge_qty * (self.hedge_entry_p - spy_p) if self.hedge_qty > 0 else 0
            
            self.account_value = self.cash + pos_mv + hedge_pnl
            self.peak_value = max(self.peak_value, self.account_value)
            
            p_ret = (self.account_value - self.prev_account_value) / self.prev_account_value if self.prev_account_value > 0 else 0
            self.portfolio_returns.append(p_ret)

        # 2. Rebalance
        day_rankings = self.rankings_np[self.current_step]
        top_k_indices = np.argpartition(day_rankings, -n_stocks)[-n_stocks:]
        top_k_scores = day_rankings[top_k_indices]
        
        # LEVEL 2/3: Sizing Logic
        temperature = 2.0
        exp_scores = np.exp((top_k_scores - np.max(top_k_scores)) / temperature)
        conviction_weights = exp_scores / np.sum(exp_scores)
        
        if self.risk_parity:
            stock_vols = self.vols_np[self.current_step][top_k_indices]
            risk_adjusted_weights = conviction_weights / (stock_vols + 1e-6)
            weights = risk_adjusted_weights / np.sum(risk_adjusted_weights)
        else:
            weights = conviction_weights
            
        # SENIOR FIX: Increase cap to 50% to allow AI to ride winners
        weights = np.clip(weights, 0.0, 0.50)
        weights = weights / np.sum(weights) # Re-normalize after clipping
        
        target_notional = self.account_value * target_lev
        
        turnover_notional = 0.0
        current_prices = self.prices_np[self.current_step]
        new_positions = {}
        for t_idx, old_qty in self.positions.items():
            if t_idx not in top_k_indices: turnover_notional += old_qty * current_prices[t_idx]
        for i, t_idx in enumerate(top_k_indices):
            p = current_prices[t_idx]
            if p > 0:
                t_qty = (target_notional * weights[i]) / p
                turnover_notional += abs(t_qty - self.positions.get(t_idx, 0.0)) * p
                new_positions[t_idx] = t_qty
        
        spy_price = self.spy.iloc[self.current_step]['close']
        t_hedge_notional = self.account_value * hedge_ratio
        new_h_qty = t_hedge_notional / spy_price if spy_price > 0 else 0
        turnover_notional += abs(new_h_qty - self.hedge_qty) * spy_price
        
        self.positions = new_positions; self.hedge_qty = new_h_qty; self.hedge_entry_p = spy_price
        # SENIOR FIX: Lower friction to 5bps
        friction_cost = turnover_notional * 0.0005
        self.account_value -= friction_cost
        pos_mv_now = sum([q * current_prices[t] for t, q in self.positions.items()])
        self.cash = self.account_value - pos_mv_now
        
        self.account_history.append(self.account_value)
        self.spy_history.append(spy_price)

        # 3. Reward
        reward = 0; done = False
        if self.current_step > 5:
            acc_ret_5d = (self.account_value / self.account_history[self.current_step - 5]) - 1.0
            spy_ret_5d = (spy_price / self.spy_history[self.current_step - 5]) - 1.0
            alpha_5d = acc_ret_5d - spy_ret_5d
            reward = alpha_5d * 100.0
            
            dd = (self.account_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
            # --- SENIOR FIX (V5.8): Loose Drawdown Threshold ---
            # Increase threshold from -3% to -10% to allow the agent to weather pullbacks.
            if dd < -0.10: reward -= abs(dd) * 5.0
            if self.account_value < self.initial_capital * 0.5: reward -= 50.0; done = True

        self.prev_account_value = self.account_value
        self.current_step += 1
        if self.current_step >= len(self.dates) - 1: done = True
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        print(f"Step: {self.current_step} | Value: ${self.account_value:,.2f} | Pos: {len(self.positions)}")
