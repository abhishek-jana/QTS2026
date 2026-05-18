# UQTS-2026: Mission Control UI - 16-Instrument Operational Manual (V2.5)

## 1. Objective
A production-grade, high-density institutional cockpit for the UQTS-2026 platform. This guide provides a detailed technical inventory of all 16 visual instruments used to monitor quad-modality neural signals and Bayesian risk health.

---

## 2. Functional Instrument Inventory

### Group 1: Spectral Physics (4 Instruments)
1.  **Price Chart**: High-resolution OHLCV candlesticks synchronized to the model's Point-in-Time (PIT) knowledge.
2.  **Wavelet Heatmap (CWT)**: Spectrogram using Morlet kernels. Brightness indicates energy density. Y-axis represents frequency (scales 1 to 128).
3.  **ADF Meter**: Real-time Augmented Dickey-Fuller p-value. Stationarity verification; target threshold is **< 0.05**.
4.  **Alpha State**: Status indicator (**READY/IDLE**) showing if the neural kernel has locked onto a valid spectral pulse.

### Group 2: Neural Logic (1 Instrument)
5.  **Neural SHAP Fusion**: Factor-based interpretation of latent weights.
    *   **Momentum (Temporal)**: Sequence-based trend memory.
    *   **Volatility (Spatial)**: Local variance and energy pulses.
    *   **Sentiment (Graph)**: Cross-sectional ticker/sector rotation influence.
    *   **Liquidity (Volume)**: Flow toxicity and volume-price divergence.

### Group 3: Metacognition (3 Instruments)
6.  **Bayesian Belief Gauge**: $P(Valid | Data)$ confidence score.
    *   **< 60%**: Automated position scaling (downsizing) triggered.
    *   **> 80%**: Regime alignment confirmed; high conviction.
7.  **Manifold Drift (t-SNE)**: 2D latent projection. Circles are training centroids; Crosses are live states. Separation indicates Out-Of-Sample (OOS) regime drift.
8.  **Cumulative Info Gain**: Line chart tracking the "realized alpha" of the session. A plateau suggests regime exhaustion.

### Group 4: Market Dynamics (3 Instruments)
9.  **Decile Ladder**: Sorting of the universe by Z-score. Provides actionable Long/Short polarities.
    *   **Flicker-Shield 2.0**: Scores are quantized to 6 decimal places and re-centered around a stable median to prevent Long/Short flip-flopping due to noise.
    *   **Relative Alpha**: Stocks are labeled relative to the universe median, ensuring a balanced long/short perspective regardless of macro drift.
10. **Sector Matrix**: Interactive grid of industrial exposure. Each block shows conviction (α) and net exposure for a specific GICS sector.
11. **L/S Equity Spread**: Tracking the cumulative return delta between Top and Bottom decile portfolios.

### Group 5: Execution Intelligence (5 Instruments)
12. **Implementation Shortfall (IS)**: Real-time efficiency tracking in Basis Points (BPS). Target is **< 5bps**.
13. **OMS Status Counters**: Counters for **FILLED**, **WORKING**, and **REJECTED** orders in the C++26 pipe.
14. **Live Order Stream**: Scrolling log of timestamped execution events (Side, Ticker, Qty, Status).
15. **Slippage Heatmap**: 5x5 dot matrix representing liquidity distribution and fill impact.
16. **Sharpe Comparison**: Real-time Sharpe Ratio delta between **Champion (V1)** and **Challenger (V2)** models.

---

## 3. Standard Operating Procedure (SOP)

### Phase 1: Screening
Use the **Sector Matrix** (10) to identify high-conviction industrial clusters. Filter the **Decile Ladder** (9) by clicking on a sector block.

### Phase 2: Verification
Select a ticker to hydrate the **Spectral Alpha** panel. Confirm that the **ADF Meter** (3) is < 0.05 and the **Wavelet Heatmap** (2) shows a stable energy pulse.

### Phase 3: Risk Audit
Check the **Metacognition** belief score (6). If it drops below 60%, audit the **Manifold Drift** (7) for signs of OOS regime shifts.

### Phase 4: Execution Review
Monitor the **OMS Counters** (13) and **Order Stream** (14). Ensure **Implementation Shortfall** (12) remains within institutional limits (< 5 BPS).
