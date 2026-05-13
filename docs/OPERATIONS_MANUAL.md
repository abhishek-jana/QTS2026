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
4.  **Race Mode (2 Weeks):** Compare the V1 (Live) and V2 (Shadow) PnL/Slippage side-by-side.
5.  **Promotion:** If V2 mathematically outperforms V1 out-of-sample for 10 consecutive trading days, swap the model binaries.

---

## 4. The 30-Day Proving Ground (Pilot Instructions)

For the first 30 days of live paper trading, the pilot must adhere to the **"Hands-Off"** protocol:

*   **Zero Code Changes:** Do not modify the model architecture or weights.
*   **Slippage Audit:** Daily comparison of the **Tuesday 3:50 PM EST Fill Price** against the **4:00 PM Official Close**.
*   **Integrity Check:** Verify the T+1 "Sealed Envelope" logic executes without manual intervention.

---
**Status:** Operational Protocol Active.
**Target:** $265k Milestone Verification.
