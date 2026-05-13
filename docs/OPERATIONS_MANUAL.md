# Institutional Operations & Maintenance Manual (V7.4.3)

This document codifies the maintenance schedule, performance triggers, and deployment protocols for the **UQTS-2026** platform to combat Concept Drift and ensure long-term Alpha stability.

## 1. Retraining Schedule (The "Oil Change")

Institutional quants do not chase weekly noise. Retraining follows a dual-horizon approach based on the specific physics of each engine.

### Alpha Engine (RankNet) :: Quarterly
*   **Frequency:** Every 3 months (aligned with Earnings Seasons).
*   **Rationale:** Market regimes fundamentally rotate as new fundamental data (earnings) is digested. Quarterly retraining allows the Variable Selection Network (VSN) to remap capital flows (e.g., rotating from Growth to Value) without overfitting to short-term volatility.

### Risk Engine (RL Pilot) :: Bi-Annually / Yearly
*   **Frequency:** Every 6–12 months.
*   **Rationale:** The physics of "Market Fear" and crashes are structural. Panics in 2026 mathematically resemble panics from 2020. Retraining too often causes the RL agent to "forget" extreme tail-risk events and overfit to recent mild pullbacks.

---

## 2. Performance-Based Triggers (The Alarms)

If any of these thresholds are breached on the **Live Paper Account**, an emergency review/retrain is mandated regardless of the schedule.

| Metric | Alarm Threshold | Required Action |
| :--- | :--- | :--- |
| **Max Drawdown** | > -22% (Relative) | Retrain **RL Pilot** (Risk sensors are misaligned). |
| **IC Decay** | Rolling 14-day IC < 0.02 | Retrain **RankNet** (Alpha model is guessing blindly). |
| **Win Rate Floor** | < 48% over 30 days | Audit **Signal Physics** and Ingestion Pipeline. |

---

## 3. The "Shadow Mode" Deployment Protocol

Never overwrite a productive model with an unproven successor. Follow the **Institutional Promotion Path**:

1.  **Develop V2:** Train the new model version on the updated DuckDB dataset.
2.  **Backtest:** Ensure V2 beats the benchmark and the current V1 in the `rl eval` suite.
3.  **Shadow Deployment:** Run V2 on a separate instance. It generates orders and telemetry, but **does not** send them to the broker.
4.  **Race Mode (2 Weeks):** Compare the V1 (Live) and V2 (Shadow) PnL/SQ side-by-side.
5.  **Promotion:** If V2 mathematically outperforms V1 out-of-sample for 10 consecutive trading days, swap the model binaries.

---

## 4. The 30-Day Proving Ground (Pilot Instructions)

For the first 30 days of live paper trading, the pilot must adhere to the **"Hands-Off"** protocol:

*   **Zero Code Changes:** Do not modify the model architecture or weights.
*   **Slippage Audit:** Daily comparison of the **Tuesday 3:50 PM EST Fill Price** against the **4:00 PM Official Close**.
*   **Integrity Check:** Verify the T+1 "Sealed Envelope" logic executes without manual intervention.

---

## 5. System Metacognition: Policy Conviction

The **Policy Conviction** metric in the Mission Control Cockpit is a dynamic "Trust Score" representing the fusion of the system's dual brains. It prevents the system from trading during macro panics or when the Alpha model loses its edge.

### The Formula: Fused Intelligence
$$\text{Policy Conviction} = \text{Bayesian Belief (Sword)} \times \text{RL Leverage (Shield)}$$

### Component A: Bayesian Belief (Model Integrity)
*   **Mechanism:** Every 3 trading days, the system compares RankNet's predicted rankings against actual realized market returns.
*   **Dynamics:** 
    *   **IC Outperformance:** If predictions match reality (Positive IC), the score drifts toward **1.0**.
    *   **Concept Drift:** If the model starts guessing blindly (Zero/Negative IC), the score decays toward **0.0**.
*   **Baseline:** Starts at **0.5** for all new deployments.

### Component B: RL Leverage (Macro Risk)
*   **Mechanism:** The RL Pilot monitors 32 macro sensors (Volatility Velocity, Trend Ratios, Drawdowns).
*   **Dynamics:** Outputs **1.0** (Active) or **0.0** (Cash/Panic).
*   **Kill-Switch:** If the Shield is up (RL=0.0), the total Conviction instantly drops to **0%**, regardless of individual stock signals.

---

## 6. Allocation Margin: The "Dark Horse" Capability

The system supports high-density allocation for testing high-conviction "winner-take-all" scenarios. This is controlled via `config.yaml`.

### The "Robust Training, Aggressive Execution" Split
Following the V7.4.3 logic audit, the configuration has been split to enforce an institutional "Train with weights, race without them" philosophy. **No manual config swapping is required between training and execution.**

Open `config.yaml` to see the decoupled logic:

#### 1. RL Training Physics (The Shield)
```yaml
rl_training_physics:
  allocation_temperature: 0.5
  max_single_asset_cap: 0.15
```
*   **Role:** Used *only* during `python run.py rl train`.
*   **Note:** This is explicitly named `rl_training_physics` to differentiate it from RankNet AI training parameters. It only affects the Risk Engine.
*   **Logic:** Forces the RL Pilot to learn in a strict, diversified environment. The pilot learns to fear drawdowns and respect friction, building a highly defensive "Shield" that aggressively pulls capital to cash during macro panics.

#### 2. Execution Muscle (The Sword)
```yaml
execution_muscle:
  allocation_temperature: 0.1
  max_position_size: 0.50
  max_single_asset_cap: 0.50
```
*   **Role:** Used during `live` trading and `rl eval` backtesting.
*   **Logic:** The 0.1 temperature transforms the execution layer into a "High-Contrast" sniper, amplifying small score gaps into massive bets up to 50% of the portfolio.

### Performance Milestone (V7.4.3 Audit)
*   **Final NLV:** **$310,130.21 (+210%)**
*   **Alpha vs SPY:** **+156.73%**
*   **Monte Carlo:** Mean path beat SPY Benchmark in 20 synthetic regimes.
*   **Risk Profile:** Drawdown reduced from **-31%** (Baseline) to **-23.5%** (RL Survivor).

*Strategic Note: The 0.1 temperature transforms the portfolio into a "High-Contrast" sniper. It rewards the #1 pick with ~80% more capital than the #2 pick, allowing the system to fully capitalize on "Winner-Take-All" market surges.*

### How to Revert
To return to professional diversified standards for execution, update the `execution_muscle` section in your `config.yaml` to match the training physics:
```yaml
execution_muscle:
  allocation_temperature: 0.5
  max_position_size: 0.15
  max_single_asset_cap: 0.15
```

---

## 7. Decoupled Operational Roles: The "Tri-Brain" Logic

To ensure system integrity and prevent logic interference, the platform operates as three independent modules connected via a "Sealed Envelope" (Redis).

### Role 1: The Analyst (RankNet AI)
*   **Focus:** Pure Signal Physics.
*   **Input:** 1.3M rows of Wavelet/Momentum data.
*   **Output:** The 60-stock Alpha Ladder.
*   **Philosophy:** "I see the edge, but I ignore the dollars."

### Role 2: The Captain (RL Pilot)
*   **Focus:** Strategy & Survival.
*   **Input:** 32 Macro Sensors + Analyst's scores.
*   **Output:** Risk Toggle (1.0x/0x) and Concentration (Top 5/12).
*   **Philosophy:** "I see the risk, but I ignore the tickers."

### Role 3: The Mechanic (Execution Bot)
*   **Focus:** Mechanical Accuracy.
*   **Input:** Weights % from Redis + Live Penny Prices.
*   **Output:** Exact Share Orders to Alpaca/IBKR.
*   **Philosophy:** "I see the orders, but I ignore the strategy."

---

## 8. Self-Healing Data Pipeline

To ensure signal integrity in volatile or illiquid stocks (e.g., TSLA), the system features an automated **Data Sanitization** layer in the `AlphaUniverse`.

*   **Gap Filling:** If the API returns NaNs (due to packet loss or trading halts), the system automatically performs a per-ticker **Forward-Fill (ffill)** followed by a **Backward-Fill (bfill)**.
*   **Physics Protection:** This ensures the Wavelet Spectrograms and Momentum sensors receive continuous numerical arrays, preventing "Black Hole" signals where a single missing bar could zero-out a stock's score for 63 days.
*   **Execution:** This occurs automatically during both `python run.py signal ingest` (Historical) and `python run.py live` (Live Snapshot).
*   **Roadmap Note:** In future versions, the `InstitutionalIngestor` can be upgraded to use **Multi-API Redundancy** (e.g., cross-referencing Tiingo with Polygon or Alpaca) to physically fill gaps with secondary source data instead of relying purely on mathematical fills.

---

## 9. Future RL Pilot Improvements (The "Lazy Bull" Patch)

Based on the V7.4.3 performance audit (comparing 700k vs 850k training steps), it was observed that the agent can become "lazy" by overfitting to the historical bull market, sacrificing Monte Carlo robustness for absolute historical PnL.

The following strategies are codified for future pilot versions:

### Strategy 1: Sync the Danger (High-Density Training)
*   **Problem:** Training with a 15% cap makes the agent "too brave" because losses from small positions aren't painful enough to trigger the Shield.
*   **Improvement:** Set `rl_training_physics.max_single_asset_cap` to **0.35 or 0.50** during training.
*   **Goal:** Force the agent to "feel the pain" of a concentrated bet during training so it learns to deploy the cash-shield early.

### Strategy 2: Punish Laziness (Reward Shaping)
*   **Problem:** High absolute returns from a bull market can drown out the penalties for drawdowns, causing the agent to stop trading (low churn).
*   **Improvement:** 
    *   Increase the **Absolute Bleed Penalty** multiplier in `rl_environment.py`.
    *   Implement a **"Survival Reward"** for being in 100% cash during periods where macro sensors (Volatility Velocity) are spiking.
*   **Goal:** Ensure the agent maintains active "Churn" (risk management) even when the overall trend is positive.

---
**Status:** Operational Protocol Active.
**Target:** $265k Milestone Verification.
