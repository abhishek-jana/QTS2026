# UQTS-2026: Phase 3 - Reinforcement Learning Pilot

Welcome to the `feat/rl-optimizer` branch. This branch represents the transition of the UQTS-2026 platform from a **Supervised Predictor** to an **Autonomous Agent**.

## 1. The Goal: From Price Prediction to Portfolio Policy
The current system (V1-V2) excels at predicting a 5-day return (IC ~0.20). However, the execution logic is still based on human-written "if/else" rules.
**Phase 3** replaces the human rules with a **Deep Reinforcement Learning (RL)** pilot that learns exactly how to size positions and apply leverage to maximize the **Sortino Ratio**.

## 2. New Components

### A. `alpha_factory/rl_environment.py` (PortfolioGym)
A custom Gymnasium-compatible environment that:
- Uses the **Challenger V2** rankings as "Sensors."
- Implements an action space for **Gross Leverage**, **Concentration**, and **Hedge Ratios**.
- Penalizes drawdowns and rewards risk-adjusted compounding.

### B. `scripts/train_rl_pilot.py`
A training harness for the **PPO (Proximal Policy Optimization)** agent. It runs the pilot through thousands of historical scenarios to learn a robust survival policy.

### C. `SimulationEngineV5` (The Ferrari)
A high-performance simulation engine that uses **Batch Inference** and **In-Memory RAM** to run 3.5 years of 15-minute data in minutes, enabling rapid RL feedback loops.

## 3. How to Train the RL Agent

1.  **Pre-compute Features**: RL training requires thousands of resets. Run this to save the model rankings to disk for instant access:
    ```bash
    python scripts/precompute_rl_data.py
    ```

2.  **Launch Training**:
    ```bash
    python scripts/train_rl_pilot.py
    ```

3.  **Evaluate**:
    The trained policy will be saved to `models/rl_pilot_final.zip`. You can then run the pilot in production mode using `run.py prod`.

---
**Status**: ACTIVE DEVELOPMENT (Phase 3)
**Author**: Gemini CLI (Expert Quant)
