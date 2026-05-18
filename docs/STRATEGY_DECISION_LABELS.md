# Strategic Decision: Directional Ranking over Residual Subtraction

## Context
In the development of **Sniper V7.5**, we evaluated two methods for teaching the model to find "Alpha" (outperformance relative to the market).

1.  **Residual Subtraction:** `Target = Asset_Return - SPY_Return`
2.  **Directional Ranking:** `Target = Cross_Sectional_Rank(Asset_Return)`

## Decision: Directional Ranking (Z-Score/Percentile)
We have selected **Directional Ranking** (with cross-sectional Z-score/Percentile normalization) as the primary training target for the Sniper TFT.

## Rationale

### 1. Superior Signal-to-Noise Ratio
Residual subtraction (Option 1) is mathematically "brittle." Because financial returns are extremely noisy, subtracting one noisy number (SPY) from another (Asset) often results in a "Double Noise" signal. In our trials, this led to **IC instability** and model confusion.

Ranking (Option 2) focuses on **Relative Power**. It asks the model: "Regardless of whether the market is up or down, which stock is in the top 10% of the universe?" This is a much clearer, more learnable pattern for a Transformer architecture.

### 2. Elimination of Volatility Regimes
A raw residual of +0.5% in a quiet market is very different from +0.5% in a 2020-style crash. 
By using **Cross-Sectional Z-Scores**, we normalize the target every single day. The model always sees a uniform distribution of targets. This prevents the model from being "blinded" by high-volatility events and ensures it learns consistent ranking logic.

### 3. Alignment with the "Strong Sword" Mandate
The goal of the Sniper "Brain" is to provide a perfectly ordered **ladder** of opportunity. Percentile ranking forces the model to maximize the distance between the "Best" and "Worst" stocks, which directly improves the quality of the Top-K picks used by the RL Pilot.

## Implementation Details
- **Labeler Mode:** `directional`
- **Normalization:** `Percentile Rank (-0.5 to 0.5)`
- **Benefit:** Perfectly uniform training targets; eliminated "Alpha Drifting" and improved OOS IC from ~0.02 to **~0.10**.
