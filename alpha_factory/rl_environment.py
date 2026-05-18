import os
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from gymnasium import spaces

from qts_core.logger import logger
from alpha_factory.observation_utils import build_rl_observation, calculate_safe_weights


class PortfolioGym(gym.Env):
    """
    Expert-Grade Reinforcement Learning Environment for V7.4 Sniper.
    """

    def __init__(
        self,
        rankings_df,
        prices_df,
        vols_df,
        spy_df,
        config_path="config.yaml",
        meta_controller=None,
    ):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.rankings = rankings_df
        self.prices = prices_df
        self.vols = vols_df
        self.spy = spy_df
        self.meta_controller = meta_controller

        self.initial_capital = 100000.0

        # Action Space: [Leverage (0-1.0), Concentration Index, Execution Trigger]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(32,), dtype=np.float32
        )

        self.dates = sorted(self.rankings.index.unique())
        self.ticker_list = [c for c in self.rankings.columns if c not in ['date']]
        self.prices_np = self.prices.values
        self.rankings_np = self.rankings.values
        self.vols_np = self.vols.values

        n = len(self.spy)
        self.spy_close_np = self.spy['close'].values.astype(np.float64)
        self.spy_vol21_np = self._safe_col(self.spy, 'vol_21', default=0.02, n=n)
        self.spy_ma_ratio_np = self._safe_col(self.spy, 'ma_ratio', default=1.0, n=n)
        self.spy_rsi14_np = self._safe_col(self.spy, 'rsi_14', default=50.0, n=n)
        self.spy_ret_np = self._safe_col(self.spy, 'ret', default=0.0, n=n)

        self.spy_vol_velocity = (
            self.spy['vol_21'].diff().rolling(5).mean().fillna(0).values
        )

        self.date_dow = np.array([d.weekday() for d in self.dates], dtype=np.float64)

        self.reset()

    @staticmethod
    def _safe_col(df, col, default, n):
        if col in df.columns:
            arr = df[col].values.astype(np.float64)
            return np.nan_to_num(arr, nan=default)
        return np.full(n, default, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.account_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}  # {ticker_idx: qty}
        self.account_history = [self.initial_capital]
        self.spy_history = [float(self.spy_close_np[0])]
        self.peak_value = self.initial_capital
        self.hedge_qty = 0.0
        self.hedge_entry_p = 0.0
        self.last_target_lev = 1.0
        self.last_n_stocks = 12
        self.steps_since_rebalance = 0

        # Adversarial Training: Pre-generate signal corruption mask
        self.signal_corruption_mask = np.ones(len(self.dates))
        num_corrupt_chunks = np.random.randint(2, 5)
        for _ in range(num_corrupt_chunks):
            start = np.random.randint(0, len(self.dates) - 20)
            self.signal_corruption_mask[start:start+10] = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        step = min(self.current_step, len(self.dates) - 1)

        scores = self.rankings_np[step].copy()
        if hasattr(self, 'signal_corruption_mask'):
            scores *= self.signal_corruption_mask[step]

        scaled_scores = scores * 100.0
        sorted_scores = np.sort(scaled_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        
        if len(top_10) < 10: top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10: bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))

        safe_nlv = max(self.account_value, 1.0)
        drawdown = (self.account_value - self.peak_value) / (self.peak_value + 1e-6)

        current_prices = self.prices_np[step]
        pos_mv = sum(qty * current_prices[t_idx] for t_idx, qty in self.positions.items())
        current_lev = pos_mv / safe_nlv

        if self.meta_controller is not None:
            try:
                belief = float(self.meta_controller.get_position_scaler()) * 100.0
            except:
                belief = float(np.mean(top_10))
        else:
            belief = float(np.mean(top_10))

        vol_21 = self.spy_vol21_np[step]
        ma_ratio = self.spy_ma_ratio_np[step]
        rsi_14 = self.spy_rsi14_np[step]
        spy_ret = self.spy_ret_np[step]
        vol_vel = self.spy_vol_velocity[step] * 1000.0
        spy_trend = (ma_ratio - 1.0) * 10.0
        dow = self.date_dow[step] / 6.0

        return build_rl_observation(
            top_10_scores=top_10,
            bot_10_scores=bot_10,
            belief=belief,
            drawdown=drawdown,
            vol_21=vol_21,
            current_lev=current_lev,
            vol_vel=vol_vel,
            spy_trend=spy_trend,
            rsi=(rsi_14 - 50.0) / 50.0,
            spy_ret=spy_ret,
            cash_ratio=self.cash / safe_nlv,
            dow=dow
        )

    def step(self, action):
        leverage_action, concentration_idx, exec_trigger = action
        if not np.all(np.isfinite(action)):
            leverage_action, concentration_idx, exec_trigger = 1.0, 1.0, 0.0

        current_prices = self.prices_np[self.current_step]
        spy_price = float(self.spy_close_np[self.current_step])

        pos_mv = sum(qty * current_prices[t_idx] for t_idx, qty in self.positions.items())
        self.account_value = self.cash + pos_mv + self.hedge_qty * (self.hedge_entry_p - spy_price)
        self.peak_value = max(self.peak_value, self.account_value)

        reward = 0.0
        turnover_notional = 0.0
        
        if self.current_step > 5:
            prev_acc = self.account_history[-1]
            prev_spy = self.spy_history[-1]
            agent_ret_1d = (self.account_value / (prev_acc + 1e-6)) - 1.0
            spy_ret_1d = (spy_price / (prev_spy + 1e-6)) - 1.0
            current_lev_ratio = pos_mv / (self.account_value + 1e-6)
            vol_vel = self.spy_vol_velocity[self.current_step]

            reward += float(agent_ret_1d - spy_ret_1d) * 3000.0
            loss_mag = max(0.0, -float(agent_ret_1d))
            reward -= (loss_mag * loss_mag) * 2500.0

            top_conviction = np.mean(np.sort(self.rankings_np[self.current_step])[-5:])
            is_signal_failing = (vol_vel > 0.0003) and (top_conviction < 0.008)

            if is_signal_failing:
                if exec_trigger > 0.5: reward -= 2.0
                if current_lev_ratio > 0.1: reward -= current_lev_ratio * 5.0
            
            if self.steps_since_rebalance > 20:
                reward -= (self.steps_since_rebalance - 20) * 0.1

        current_dt = self.dates[self.current_step]
        should_rebalance = (exec_trigger > 0.7) or (current_dt.weekday() == 0)

        if should_rebalance:
            self.steps_since_rebalance = 0
            if self.meta_controller is not None:
                prev_scores = self.rankings_np[self.current_step - 1] if self.current_step > 0 else self.rankings_np[0]
                prev_prices = self.prices_np[self.current_step - 1] if self.current_step > 0 else self.prices_np[0]
                real_rets_for_mc = np.where(prev_prices > 1e-6, (current_prices / prev_prices) - 1.0, 0.0)
                self.meta_controller.update_belief(real_rets_for_mc, prev_scores)

            self.last_target_lev = float(np.clip(leverage_action, 0.0, 1.0))
            self.last_n_stocks = [5, 10, 15, 20][int(np.clip(concentration_idx, 0, 3.99))]

            scores = self.rankings_np[self.current_step]
            temp = self.config.get('rl_training_physics', {}).get('allocation_temperature', 0.5)
            asset_cap = self.config.get('rl_training_physics', {}).get('max_single_asset_cap', 0.15)

            top_k_indices, weights = calculate_safe_weights(scores, self.last_n_stocks, asset_cap, temp)

            target_notion = self.account_value * self.last_target_lev
            new_positions = {}
            for t_idx, old_qty in self.positions.items():
                if t_idx not in top_k_indices:
                    turnover_notional += old_qty * current_prices[t_idx]

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
            reb_cost = turnover_notional * 0.0005
            self.account_value -= reb_cost
            if self.current_step > 5:
                reward -= (turnover_notional / (self.account_value + 1e-6)) * 1.5

        pos_mv_now = sum(q * current_prices[t] for t, q in self.positions.items())
        self.cash = self.account_value - pos_mv_now
        
        self.account_history.append(self.account_value)
        self.spy_history.append(spy_price)

        self.current_step += 1
        done = (self.current_step >= len(self.dates) - 1 or self.account_value < self.initial_capital * 0.3)
        reward = np.clip(np.nan_to_num(reward, nan=-1.0), -20.0, 20.0)
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        logger.info(f"Step: {self.current_step} | Value: ${self.account_value:,.2f} | Pos: {len(self.positions)}")
