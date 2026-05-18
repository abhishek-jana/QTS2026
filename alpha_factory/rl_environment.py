import os
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from gymnasium import spaces

from qts_core.logger import logger


class PortfolioGym(gym.Env):
    """
    Expert-Grade Reinforcement Learning Environment for V7.4 Sniper.

    Features:
      - Smart Binary Risk Toggle (100% On / 100% Cash)
      - Shield & Sword Reward Logic (defensive alpha + offensive alpha)
      - Volatility Velocity & SPY Trend Sensors
      - Institutional Friction & Turnover Constraints
      - Optional BayesianMetaController belief feed (real metacognition)

    Efficiency notes:
      - SPY DataFrame columns are pre-extracted into numpy arrays at __init__,
        so the per-step _get_obs() avoids pandas .iloc (~10-50x slower than
        plain numpy indexing). Originally each obs incurred 4-5 pandas row
        accesses; with 1500 steps x 12 parallel envs x many epochs this was
        a measurable fraction of training wall-clock.
      - The dow lookup uses precomputed weekday integers.
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
        # Optional metacontroller; if provided, its belief replaces the
        # raw-conviction proxy in the observation vector.
        self.meta_controller = meta_controller

        self.initial_capital = 100000.0

        # Action Space: [Leverage (0-1.0), Concentration Index, Execution Trigger]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space: 32 macro/micro sensors
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(32,), dtype=np.float32
        )

        self.dates = sorted(self.rankings.index.unique())
        self.ticker_list = [c for c in self.rankings.columns if c not in ['date']]
        self.prices_np = self.prices.values
        self.rankings_np = self.rankings.values
        self.vols_np = self.vols.values

        # EFFICIENCY (Fix #9): pre-extract SPY columns into 1-D numpy arrays
        # so the hot loop doesn't pay pandas iloc/get overhead per step.
        n = len(self.spy)
        self.spy_close_np = self.spy['close'].values.astype(np.float64)
        self.spy_vol21_np = self._safe_col(self.spy, 'vol_21', default=0.02, n=n)
        self.spy_ma_ratio_np = self._safe_col(self.spy, 'ma_ratio', default=1.0, n=n)
        self.spy_rsi14_np = self._safe_col(self.spy, 'rsi_14', default=50.0, n=n)
        self.spy_ret_np = self._safe_col(self.spy, 'ret', default=0.0, n=n)

        # Pre-calculate Volatility Velocity
        self.spy_vol_velocity = (
            self.spy['vol_21'].diff().rolling(5).mean().fillna(0).values
        )

        # Pre-compute weekday integers for date-of-week feature
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
        return self._get_obs(), {}

    def _get_obs(self):
        step = min(self.current_step, len(self.dates) - 1)

        scores = self.rankings_np[step]
        scaled_scores = scores * 100.0
        sorted_scores = np.sort(scaled_scores)
        top_10 = sorted_scores[-10:][::-1]
        bot_10 = sorted_scores[:10]
        if len(top_10) < 10:
            top_10 = np.pad(top_10, (0, 10 - len(top_10)))
        if len(bot_10) < 10:
            bot_10 = np.pad(bot_10, (0, 10 - len(bot_10)))

        safe_nlv = max(self.account_value, 1.0)
        drawdown = (self.account_value - self.peak_value) / (self.peak_value + 1e-6)

        current_prices = self.prices_np[step]
        pos_mv = sum(
            qty * current_prices[t_idx] for t_idx, qty in self.positions.items()
        )
        current_lev = pos_mv / safe_nlv

        # BELIEF: prefer real metacontroller belief if wired, else proxy.
        if self.meta_controller is not None:
            try:
                belief = float(self.meta_controller.get_position_scaler()) * 100.0
            except Exception as e:
                logger.warning(f"meta_controller belief read failed: {e}; falling back to proxy.")
                belief = float(np.mean(top_10))
        else:
            belief = float(np.mean(top_10))

        # Fast numpy lookups (no pandas overhead)
        vol_21 = self.spy_vol21_np[step]
        ma_ratio = self.spy_ma_ratio_np[step]
        rsi_14 = self.spy_rsi14_np[step]
        spy_ret = self.spy_ret_np[step]

        vol_vel = self.spy_vol_velocity[step] * 1000.0
        spy_trend = (ma_ratio - 1.0) * 10.0
        dow = self.date_dow[step] / 6.0

        obs = np.concatenate([
            top_10, bot_10,
            [belief, drawdown, vol_21, current_lev],
            [vol_vel, spy_trend, (rsi_14 - 50.0) / 50.0, spy_ret],
            [self.cash / safe_nlv, 0.0, 1.0, dow],
        ]).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        # Action space mapping: [Leverage (0-1.2), Concentration, Rebalance Trigger]
        leverage_action, concentration_idx, exec_trigger = action
        if not np.all(np.isfinite(action)):
            leverage_action, concentration_idx, exec_trigger = 1.0, 1.0, 0.0

        # -------------------------------------------------------------------
        # 1. MARK-TO-MARKET (ADVANCE TO TODAY'S REALITY)
        # Advance prices to the current step (t). These prices represent the
        # reality at the end of the step, after the positions decided in the
        # *previous* step have been held.
        # -------------------------------------------------------------------
        current_prices = self.prices_np[self.current_step]
        spy_price = float(self.spy_close_np[self.current_step])

        # Value of positions decided YESTERDAY (at t-1) marked at TODAY'S prices
        pos_mv = sum(
            qty * current_prices[t_idx] for t_idx, qty in self.positions.items()
        )
        self.account_value = (
            self.cash + pos_mv + self.hedge_qty * (self.hedge_entry_p - spy_price)
        )
        self.peak_value = max(self.peak_value, self.account_value)

        # -------------------------------------------------------------------
        # 2. COMPUTE REWARD FROM REALIZED RETURN
        # Reward is based on how the allocation from (t-1) performed at (t).
        # -------------------------------------------------------------------
        reward = 0.0
        turnover_notional = 0.0
        if self.current_step > 5:
            prev_acc = self.account_history[-1]
            prev_spy = self.spy_history[-1]
            agent_ret_1d = (self.account_value / (prev_acc + 1e-6)) - 1.0
            spy_ret_1d = (spy_price / (prev_spy + 1e-6)) - 1.0

            current_lev = pos_mv / (self.account_value + 1e-6)
            vol_vel = self.spy_vol_velocity[self.current_step]

            # --- ABSOLUTE ALPHA (Unconditional) ---
            alpha_1d = agent_ret_1d - spy_ret_1d
            reward += float(alpha_1d) * 3000.0

            # --- SMOOTH ABSOLUTE BLEED PENALTY ---
            loss_mag = max(0.0, -float(agent_ret_1d))
            reward -= (loss_mag * loss_mag) * 2500.0

            # --- SMART SIGNAL GATE ---
            top_conviction = np.mean(np.sort(self.rankings_np[self.current_step])[-5:])
            is_signal_failing = (vol_vel > 0.0003) and (top_conviction < 0.008)

            if is_signal_failing:
                if exec_trigger > 0.5:
                    reward -= 2.0
                if current_lev > 0.1:
                    reward -= current_lev * 5.0

            if (
                current_lev < 0.10
                and top_conviction > 0.015
                and not is_signal_failing
                and spy_ret_1d > 0
            ):
                reward -= 2.5

        # -------------------------------------------------------------------
        # 3. CHOOSE ALLOCATION FOR THE NEXT INTERVAL (TODAY -> TOMORROW)
        # -------------------------------------------------------------------
        current_dt = self.dates[self.current_step]
        should_rebalance = (exec_trigger > 0.7) or (current_dt.weekday() == 0)

        if should_rebalance:
            # SENIOR FIX (Metacognition): Update MetaController so belief sensor is live
            if self.meta_controller is not None:
                prev_scores = self.rankings_np[self.current_step - 1] if self.current_step > 0 else self.rankings_np[0]
                prev_prices = self.prices_np[self.current_step - 1] if self.current_step > 0 else self.prices_np[0]
                real_rets_for_mc = np.where(prev_prices > 1e-6, (current_prices / prev_prices) - 1.0, 0.0)
                self.meta_controller.update_belief(real_rets_for_mc, prev_scores)

            # Continuous leverage mapping (Capped at 1.0 to prevent borrowing)
            self.last_target_lev = float(np.clip(leverage_action, 0.0, 1.0))
            # HIGH-OCTANE FIX: Allow agent to concentrate down to 5 stocks
            self.last_n_stocks = [5, 10, 15, 20][int(np.clip(concentration_idx, 0, 3.99))]

            scores = self.rankings_np[self.current_step]
            top_k_indices = np.argsort(scores)[-self.last_n_stocks:][::-1]

            temp = self.config.get('rl_training_physics', {}).get('allocation_temperature', 0.5)
            asset_cap = self.config.get('rl_training_physics', {}).get('max_single_asset_cap', 0.15)

            top_scores = scores[top_k_indices]
            exp_scores = np.exp((top_scores - np.max(top_scores)) / temp)
            weights = exp_scores / (np.sum(exp_scores) + 1e-9)

            if np.max(weights) > asset_cap:
                weights = np.clip(weights, 0, asset_cap)
                weights = weights / (np.sum(weights) + 1e-9)

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
            # Turnover tax applies to TODAY'S rebalance, deducted from account value
            reb_cost = turnover_notional * 0.0005
            self.account_value -= reb_cost
            if self.current_step > 5:
                reward -= (turnover_notional / (self.account_value + 1e-6)) * 5.0

        pos_mv_now = sum(q * current_prices[t] for t, q in self.positions.items())
        self.cash = self.account_value - pos_mv_now
        
        self.account_history.append(self.account_value)
        self.spy_history.append(spy_price)

        self.current_step += 1
        done = (
            self.current_step >= len(self.dates) - 1
            or self.account_value < self.initial_capital * 0.3
        )
        reward = np.clip(np.nan_to_num(reward, nan=-1.0), -20.0, 20.0)
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        logger.info(
            f"Step: {self.current_step} | Value: ${self.account_value:,.2f} "
            f"| Pos: {len(self.positions)}"
        )
