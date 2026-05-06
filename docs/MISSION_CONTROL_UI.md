# UQTS-2026: Mission Control UI (V2 Institutional)

## 1. Objective
A production-grade, high-density institutional cockpit for the UQTS-2026 quantitative platform. This interface provides real-time transparency into quad-modality neural signals, Bayesian risk scaling, and automated execution.

## 2. Technical Stack
- **Backend**: FastAPI (Python) + Redis Pub/Sub multiplexing.
- **Frontend**: React (Vite) + Tailwind CSS + Lightweight Charts + Recharts.
- **Protocol**: Low-latency WebSockets for 100-ticker bundle streaming.

## 3. Core Functional Sections
### 1. Spectral Alpha (Physicist's View)
- **Price Chart**: High-resolution OHLCV candles synchronized to Point-in-Time Knowledge time.
- **Wavelet Spectrogram**: Real-time Morlet CWT heatmap visualization showing multi-resolution energy states.
- **Statistical Integrity**: Live ADF (Augmented Dickey-Fuller) p-values for stationarity verification.
- **SHAP Fusion**: Neural modality weights mapped to human-readable factors (Momentum, Volatility, Sentiment, Liquidity).

### 2. Metacognition (Risk Health)
- **Bayesian Belief Gauge**: Confidence metric $P(Valid | Result)$ derived from realized alpha.
- **Manifold Drift (t-SNE)**: 2D projection of live market regimes vs. training clusters.
- **Alpha Decay Signal**: Interactive cumulative information gain curve.

### 3. Ranking Ladder & Sector Matrix
- **Decile Ladder**: Cross-sectional sorted rankings (Long/Short) with live price sync.
- **Sector Intelligence (Interactive)**:
    - Advanced aggregation: Exposure %, Ticker count, and Conviction (α) per sector.
    - **Drill-down**: Clicking a sector filters the main Ranking Ladder instantly.
    - **Focus**: Clicking a ticker hydrates the Spectral Alpha panel with high-res telemetry.

### 4. Execution Muscle (Reality Check)
- **Implementation Shortfall (IS)**: Real-time efficiency tracking in BPS.
- **OMS Queue**: Status of filled vs. working orders in the C++26 execution pipe.
- **Slippage Heatmap**: Stretched visualization of liquidity distribution and fill impact.
- **Latency Monitor**: μs-level telemetry from the Direct OSQP Context.

### 5. Research Pipeline Control
- **Champion vs Challenger**: A/B backtest telemetry showing Sharpe Ratio delta.
- **System Status**: Live terminal for training progress and WFO status.

## 4. Operational Instructions
- **Filter**: Use the Sector Matrix blocks to isolate industrial alpha regimes.
- **Analyze**: Click tickers to verify the signal physics via Wavelet and SHAP fusion.
- **Scale**: Monitor Bayesian Belief; values < 60% indicate automated position scaling.
- **Kill Switch**: Liquidation protocol active via the global header button.
