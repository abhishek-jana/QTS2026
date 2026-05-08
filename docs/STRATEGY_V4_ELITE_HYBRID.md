# Strategy V4: The Elite Hybrid (Institutional Grade)

## 1. Executive Summary
The **Elite Hybrid V4** represents the production-ready evolution of the UQTS-2026 platform. While previous versions focused on raw alpha (V3: High-Octane) or risk aversion (V1), V4 is designed for institutional capital allocation. It leverages the elite predictive power of the Challenger V2 model (IC ~0.20) while implementing professional-grade risk armor.

## 2. Core Architectural Pillars

### A. Intelligent Diversification (The "Rule of 12")
- **Concentration**: Top 12 tickers from the Decile Ladder.
- **Rationale**: Moving from 5 to 12 stocks reduces "idiosyncratic drag" (the risk that one bad earnings report wipes out the portfolio) while still maintaining high exposure to the model's alpha.

### B. Dynamic Beta Hedging
- **Mechanism**: The strategy maintains a varying short position in $SPY or $QQQ.
- **Scaling**: Hedge ratio is determined by the **Bayesian Belief Score**. 
    - *High Belief (>0.80)*: 10% Hedge (Aggressive Long).
    - *Low Belief (<0.40)*: 60% Hedge (Market Neutral / Defensive).
- **Goal**: Protect the $300k+ account value from market-wide "Gaps" and systemic crashes.

### C. The Dual-Layer Stop-Loss
1.  **Logical Stop (Bayesian)**: Automated liquidation if model conviction drops below **0.15**.
2.  **Physical Stop (Trailing)**: A hard **-5.0% Trailing Stop-Loss** per position, executed via the C++ `execution_muscle`. This protects against "Black Swan" events that occur between 15-minute inference cycles.

### D. Regime-Adaptive Leverage
- **Range**: 0.8x to 1.5x Gross Exposure.
- **Multiplier**: `Leverage = 0.8 + (Belief * 0.7)`.
- **Result**: Unlike V3's fixed 2.0x, V4 only leverages up when the "Metacognition" confirms the market regime is highly predictable.

## 3. Performance Targets (2023-2026 Projection)
| Metric | High-Octane (V3) | **Elite Hybrid (V4)** |
| :--- | :--- | :--- |
| **Total Return** | +223.81% | **+115% - 145%** |
| **Max Drawdown** | -24.5% | **-8.5%** |
| **Sharpe Ratio** | 1.85 | **2.65** |
| **Capacity** | $1M - $5M | **$50M+** |

## 4. Implementation Roadmap
1.  **Refine OMS**: Update `oms.py` to support dynamic short-hedging of index ETFs.
2.  **Trailing Logic**: Implement the `-5%` hard-exit in `execution_pipeline.hpp`.
3.  **Beta-Adjuster**: Integrate the SPY/QQQ correlation matrix into the `StrategyEngine`.

---
**Status**: DRAFT (Expert Quant Recommended)
**Date**: May 8, 2026
