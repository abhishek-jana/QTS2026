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

## CLI Usage

- `python run.py --ingest`: Ingest historical data (2016-Present)
- `python run.py --train`: Train the Supervised RankNet (2018-2022)
- `python run.py sim`: Run the Audited V5 Simulation (2023-2026)
- `python run.py rl-train`: Train the Phase 3 RL Portfolio Pilot
- `python run.py prod`: Launch the Production Inference Worker
