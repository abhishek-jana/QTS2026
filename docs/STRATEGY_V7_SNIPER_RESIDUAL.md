# STRATEGY SPEC: THE SNIPER (V7.0 RESIDUAL ALPHA)

## 1. Executive Summary
The "Sniper" strategy is a professional-grade simplification of the "Master Chief" architecture. It transitions from a complex, data-hungry Transformer/RL stack to a high-conviction, **Daily Residual Alpha** model. This strategy is specifically optimized for a high-liquidity, 60-stock universe (The "Institutional Core") and is designed to structurally outperform the S&P 500 (SPY).

## 2. Core Philosophy: Why this beats the market
The primary challenge in quant trading is separating **Alpha** (individual stock skill) from **Beta** (market noise). 

### A. Residual Alpha Targeting
*   **The Old Way:** Predicting if a stock goes up. (Result: You just track the market).
*   **The Sniper Way:** Predicting `Stock_Return - SPY_Return`. 
*   **Why:** By subtracting the market return, we force the model to ignore macro noise (interest rates, Fed news) and focus purely on why one stock is outperforming its peers. This is how top-tier firms like Citadel achieve market-neutral-style outperformance.

### B. Daily Continuous Execution (vs. Weekly)
*   **The Advantage:** Alpha decays. A signal found on Tuesday is often gone by Friday.
*   **The Math:** According to the *Fundamental Law of Active Management*, your Information Ratio (IR) is a function of your skill (IC) and your breadth (frequency). Moving from 52 trades/year to 252 trades/year mathematically increases your chances of beating the market.

## 3. The "Sniper" Stack

### I. Feature Layer: Signal Physics
*   **Fractional Differentiation:** Removes price trends while preserving memory (Stationarity without losing signal).
*   **Market Spectrograms (Wavelets):** Captures multi-scale volatility patterns that standard indicators miss.

### II. Prediction Layer: Residual GRU
*   **Model:** 3-Layer Gated Recurrent Unit (GRU).
*   **Rationale:** Transformers require massive datasets (1000+ stocks). For a 60-stock universe, a GRU is "right-sized." It is less prone to overfitting and much better at capturing the 1-5 day momentum/mean-reversion loops prevalent in large-cap stocks.

### III. Risk Layer: Inverse-Volatility Sizing
*   **The Logic:** $10k of NVDA is not the same as $10k of PG. NVDA has 3x the "risk budget."
*   **The Rule:** Position sizes are scaled by their 20-day rolling volatility. Every stock in the portfolio is calibrated to contribute the **exact same amount of risk**, creating a "smooth" equity curve that can weather market turbulence.

## 4. Implementation Roadmap

1.  **Labeling:** Update `alpha_labeler.py` to calculate `Close_T1 / Close_T0 - SPY_T1 / SPY_T0`.
2.  **Model:** Implement `alpha_ranker_sniper.py` using a Residual GRU architecture.
3.  **Training:** Focus on 15-minute bars to capture intra-day "Smart Money" flows.
4.  **Allocation:** Replace the RL policy with a deterministic **Risk-Parity Allocator** to ensure transparent, math-based sizing.

## 5. Conclusion
The Sniper strategy moves away from "AI Hype" and toward **Institutional Rigor**. It trades more often, targets specific outperformance, and manages risk through mathematical laws rather than black-box guesses.

**Status:** Ready for Implementation.
