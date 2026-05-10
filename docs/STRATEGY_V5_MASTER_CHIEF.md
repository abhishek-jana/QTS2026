# Strategy V5: "Master Chief" Architecture

## Overview
The "Master Chief" architecture (V5.3 / V5.4 / V5.5) is an institutional-grade Quantitative Trading System that combines a highly optimized Multi-Modal Neural Network (The Scout) with a Reinforcement Learning Macro Allocator (The General).

## 🧠 The Scout: RankNet Alpha Generator (0.21+ IC)
The alpha generation pipeline has been heavily upgraded to break through the 0.68 loss plateau and achieve an unprecedented Information Coefficient (IC) of >0.21.

### Key Power-Ups (V5.3 - V5.5)
- **Modality Expansion (`x_momentum`):** Added explicit 10, 20, and 60-day raw returns to bypass fractional differentiation limitations and allow the model to see pure momentum regimes (e.g., the 2024 Tech Rally). Fractional differentiation `d_param` reduced to `0.0`.
- **Autonomous Regime Discovery (V5.5):** Replaced **Residualized Returns** with **Cross-Sectional Z-Scored Returns**. Previously, residualizing ($y - \beta x$) introduced an anti-beta bias that forced the model to ignore market-wide rallies. By using simple cross-sectional rankings, the AI is free to autonomously learn that tech momentum is a winning feature in the 2024 regime.
- **Normalized Spatial Features:** Wavelet spectrograms now utilize per-scale Z-Score Normalization to stabilize standard deviations and prevent Vision Transformer (ViT) gradient saturation.
- **Spatial Feature Dropout:** Added `SpatialDropout(p=0.3)` to force the ViT to discover high-frequency intraday alpha instead of lazily relying on low-frequency macro trends.
- **Magnitude-Weighted Loss:** The `PairwiseRankLoss` multiplies its target output by the absolute difference in returns between two assets. The model focuses heavily on correctly ranking massive outliers rather than getting distracted by 1% noise.
- **Regime-Aware Temporal Decay:** Applies an exponential decay weight (`lambda = 0.0025`) anchored to the end of the training set. This forces the model to prioritize recent (e.g., 2023) market regimes over outdated (e.g., 2019) dynamics.
- **Optimized Convergence:** Switched to `AdamW` optimizer (with weight decay) and `ReduceLROnPlateau` scheduler. Automatically halves the learning rate when loss plateaus to squeeze out the deepest possible alpha before early stopping.

## 🤖 The General: RL Meta-Controller (32-Sensors)
The RL Agent controls gross exposure, cash hedging, and concentration. It views the market through a 32-sensor vector.

- **Look-Ahead Prevention (Ghost Protocol):** All macro sensors and short market value calculations are strictly shifted to `T-1` (Yesterday's close) to prevent any look-ahead bias during simulation.
- **Rolling 5-Day Reward:** The agent receives a reward proportional to the 5-day rolling alpha (Strategy Return - SPY Return), incentivizing smooth outperformance rather than daily jitter optimization.

## ⚖️ Position Sizing: Conviction & Risk Parity
The execution engines (`PortfolioGym`, `SimulationEngineV5`, `MonteCarloStressTest`) dynamically size individual stock positions using a multi-level hierarchy:

### Level 2: Conviction Weighting (Temperature-Scaled Softmax)
- Capital is allocated proportionally to the RankNet's confidence score.
- Uses `Temperature = 2.0` to flatten extreme AI confidence spikes.
- Applies a **50% Hard Cap** (increased from 35% in V5.5) per stock to allow the AI to ride winning compounders more aggressively.

### Level 3: Risk Parity / Inverse Volatility (ACTIVE)
- Configurable via `risk_parity_sizing: true` in `config.yaml`.
- The Conviction Weight is divided by the stock's trailing 21-day rolling volatility (`weight = conviction / (vol + 1e-6)`).
- **Result:** High-conviction, low-volatility stocks receive optimal capital allocation. High-conviction but dangerously volatile stocks are safely scaled down.

## 🛡️ The Governor: Deterministic Risk Overlay
- **Role:** Hard-coded Safety Fail-safe.
- **Implementation:** Non-AI, pure mathematical logic sitting between the RL Agent and the Execution Engine.
- **Rules:**
    - **Max Gross Exposure:** Hard cap at 1.0x (Cash Account).
    - **Circuit Breaker:** If portfolio drawdown > 3% intraday, force liquidation to cash.
    - **Institutional Friction (V5.5):** Reduced to **5bps (0.0005)** per trade to match institutional execution reality and prevent capital erosion during rebalancing.
