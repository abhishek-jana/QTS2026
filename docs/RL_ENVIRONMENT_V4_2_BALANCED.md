# RL Environment V4.2: "Balanced Sharpe" Configuration

## Overview
RL Environment V4.2 represents a shift from "High-Octane Aggressive Alpha" to an institutional "Balanced Sharpe" profile. It addresses action-space mapping inefficiencies and introduces professional risk-budgeting rewards.

## Key Upgrades

### 1. Action Space: Even Bucketing
- **Problem:** In V4.1, the continuous-to-discrete mapping `[2, 5, 12][int(np.clip(idx, 0, 2))]` created a "flat landscape" where the 12-stock option was mathematically under-represented compared to the 2 and 5-stock options.
- **Fix:** Expanded the action space to `[0, 3.0]` and implemented even bucketing:
    - `[0, 1.0) -> 2 Stocks` (Concentrated Alpha)
    - `[1, 2.0) -> 5 Stocks` (Standard)
    - `[2, 3.0] -> 12 Stocks` (Diversified/Defensive)
- **Impact:** The agent now has an equal probability of exploring and learning the benefit of each concentration level.

### 2. Reward Function: Institutional Rebalancing
V4.2 replaces the "Winner's Bias" with a "Risk-Adjusted Bias."

| Metric | V4.1 (Old) | V4.2 (New) | Rationale |
| :--- | :--- | :--- | :--- |
| **Alpha Reward** | `alpha * 30.0` | `alpha * 20.0` | Lower weight on raw gains. |
| **Alpha Penalty** | `alpha * 10.0` | `alpha * 40.0` | 2x penalty on underperformance to force safety. |
| **Drawdown** | Static `-0.1` at -12% | Quadratic: `-abs(dd) * 5.0` | Early and escalating pain for losses > 5%. |
| **Volatility** | None | `-abs(daily_ret) * 2.0` | Penalizes "lucky" high-variance gambles. |
| **Leverage Bonus** | Always on | Conditional on Alpha > 0 | Agent only gets "points" for leverage if it's working. |

## Expected Behavior Shift
The V4.2 agent is expected to:
1.  **Diversify more frequently:** Using the 12-stock option to "hide" from volatility penalties in choppy markets.
2.  **Modulate Leverage:** Scaling down exposure rapidly when drawdowns approach -5% to avoid the quadratic penalty.
3.  **Prioritize Sharpe:** Seeking the highest return-per-unit-of-risk rather than absolute highest return.

## Implementation Details
- **Location:** `alpha_factory/rl_environment.py`
- **Model Revision:** To be saved as `models/rl_pilot_v4_2.zip` after training.
