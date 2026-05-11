## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses canonical triage label names (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

## Strategic Logics

### Logic V1: Institutional Restrictive (DEPRECATED)
- **Goal:** Risk-Averse institutional growth.
- **Parameters:** Threshold 0.65, Multiplier 1.5x, Shorting Enabled.
- **Outcome:** -7.8% (Missed rallies due to paranoia, lost money on shorts).

### Logic V2: Unleashed (BASELINE)
- **Goal:** Beat S&P 500 with institutional safety.
- **Parameters:** 
  - `min_belief_threshold`: 0.30
  - `leverage_multiplier`: 1.5x
  - `shorting`: Disabled (Long-Only)
  - `rebalance_cadence`: Weekly (Mondays)
  - `concentration`: Top 10 stocks
- **Outcome:** +64.62% (2023-2026). Stabilized the account and captured the AI rally.

### Logic V3: High-Octane (EXPERIMENTAL)
- **Goal:** 200% - 300% Aggressive Alpha.
- **Proposed Parameters:**
  - `min_belief_threshold`: 0.15 (Extreme persistence)
  - `leverage_cap`: 2.5x (Reg-T Margin limit)
  -Concentration: Top 5 stocks (High conviction only)
  - `sizing`: Fixed 2.0x leverage (scaling only as kill-switch)
- **Status:** Validated (+223% projection).

### Logic V4: Elite Hybrid (PROPOSED)
- **Goal:** Institutional Grade Sharpe.
- **Parameters:** Top 12 stocks, Dynamic Beta Hedging, Hard Trailing Stop-Loss.

### Logic V5: Master Chief (ACTIVE)
- **Goal:** Multi-Regime Dominance via HRL-MoE and Multi-Modal Alpha.
- **Architecture:** 32-Sensor RL Agent (The General) + Multi-Modal RankNet with 0.21+ IC (The Scout).
- **Positioning:** Level 3 Risk Parity (Conviction Softmax / Rolling Volatility).
- **Spec:** See `docs/STRATEGY_V5_MASTER_CHIEF.md`.

## CLI Usage

### 1. Signal Pipeline (The Stock Picker)
- `python run.py signal ingest`: Ingest historical data for the full universe.
- `python run.py signal train`: Train the Supervised Ghost Protocol Transformer on 5-year historical data.
- `python run.py signal eval`: Compare Signal performance against baselines.
- `python run.py signal test-subset`: Run rapid verification on SPY, NVDA, and TSM.

### 2. RL Pipeline (The Macro Allocator)
- `python run.py rl data`: Pre-compute 32-sensor training data for RL.
- `python run.py rl train --steps 500000`: Train the RL Portfolio Pilot.
- `python run.py rl eval`: Run unified evaluation (Logic Audit, SPY Sim, Monte Carlo).

### 3. Execution & Operations
- `python run.py prod`: Launch the Production Inference Worker (Real Money).
- `python run.py live`: Launch the Paper Trading Worker (Simulated Money).
- `python run.py ui`: Launch the Mission Control Cockpit (FastAPI backend).
