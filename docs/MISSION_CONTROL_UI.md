# UQTS-2026: Mission Control UI - Operator's Guide (V2.5)

## 1. Objective
The UQTS-2026 Mission Control is a high-density institutional cockpit designed for real-time monitoring of quad-modality neural alpha signals. It prioritizes "Physics over Finance," providing transparency into the spectral decomposition and metacognitive state of the strategy.

## 2. Core Functional Sections

### 2.1 Spectral Alpha Telemetry (The Physicist's View)
*   **Price Chart**: High-resolution OHLCV candles synchronized to Point-in-Time Knowledge time. This is the "Ground Truth" for alpha signals.
*   **Wavelet Spectrogram**: Real-time Morlet Continuous Wavelet Transform (CWT) heatmap. 
    *   **Y-Axis**: Frequency scale (Log-spaced).
    *   **X-Axis**: Time (Rolling window).
    *   **Brightness/Color**: Energy density. High energy at lower scales (high frequency) often precedes volatility breakouts or regime shifts.
*   **Statistical Integrity (ADF)**: Live Augmented Dickey-Fuller p-values. Values **MUST be < 0.05** for stationarity verification. Signals with high p-values are considered "drifting" and carry high risk.

### 2.2 Neural Interpretability (SHAP Fusion)
*   **Momentum (Temporal)**: Weight assigned to historical price trend and sequence memory.
*   **Volatility (Spatial)**: Weight assigned to variance structures and local energy pulses.
*   **Sentiment (Graph)**: Weight assigned to cross-sectional ticker relationships and sector rotation signals.
*   **Liquidity (Volume)**: Weight assigned to volume-price divergence and flow toxicity.

### 2.3 Metacognition & Risk (Metasurface)
*   **Bayesian Belief Gauge**: A real-time confidence score $P(Valid | Data)$ derived from realized alpha vs. predicted distribution.
    *   **> 80%**: High conviction; standard sizing.
    *   **60% - 80%**: Neutral; caution advised.
    *   **< 60%**: **Automated position scaling triggered.**
*   **Manifold Drift (t-SNE)**: 2D projection of the live market manifold.
    *   **Base Dots (Circles)**: Training set centroids.
    *   **Live Dots (Crosses)**: Current market state.
    *   **Interpretation**: Large distances indicate the model is encountering OOS (Out-Of-Sample) regimes.

### 2.4 Execution & Reality Check
*   **Implementation Shortfall (IS)**: Real-time efficiency tracking in Basis Points (BPS). Measures the distance between arrival price and fill price. High IS (> 5bps) indicates toxic flow.
*   **OMS Queue**: Live status of the Order Management System.
    *   **FILLED**: Order executed and committed to the ledger.
    *   **WORKING**: Active order in the market (Iceberg/TWAP).
    *   **REJECTED**: Order blocked by risk-limit breaches (e.g., Notional/Concentration).

## 3. Standard Operating Procedure (SOP)

### Phase 1: Industrial Screening
1.  Scan the **Sector Matrix** for clusters with positive cumulative exposure and high ticker counts.
2.  Identify "Industrial Hotspots" where conviction (α) is concentrated.

### Phase 2: Signal Verification
1.  Click a ticker in the **Ranking Ladder** to hydrate the Spectral panel.
2.  Verify the **Wavelet Physics**: Does the signal show stable energy at the target horizon?
3.  Check **Neural SHAP**: Does the model's rationale (e.g., Momentum-heavy) match the visual chart structure?

### Phase 3: Risk Assessment
1.  Monitor the **Metacognition** belief score. Ensure it remains above the 60% liquidation threshold.
2.  Review the **Manifold Drift**. If crosses are drifting far from circles, tighten stop-losses.

### Phase 4: Execution Audit
1.  Verify the **OMS Queue**. Check for any REJECTED orders that might indicate system bottlenecks.
2.  Audit **Slippage Heatmaps** to ensure fills are occurring within expected liquidity pockets.

## 4. Emergency Protocol
*   **INTERFACE OVERRIDE (Header Button)**: Immediately terminates all active WebSocket streams and triggers the server-side liquidation logic. Use ONLY in cases of extreme manifold drift or execution instability.
