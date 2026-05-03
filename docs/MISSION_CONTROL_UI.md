# UQTS-2026: Mission Control UI Implementation Plan

## 1. Objective
Build a production-grade, high-density "Mission Control" cockpit for the UQTS-2026 quantitative platform. The UI must provide transparency into the "Black Box" of the ML models, monitor real-time signal physics, track execution reality, and manage the research pipeline. The interface must be zero-lag, actionable, and adhere to a dark-themed, monospace aesthetic suitable for a Senior Quant.

## 2. Architectural Decisions
- **Stack**: FastAPI (Python) backend + React (Vite) / Tailwind CSS frontend.
- **Communication**: WebSockets for low-latency, real-time streaming of metrics (CWT scrolling, belief updates, execution tracking).
- **Data Source**: A background async worker in FastAPI will simulate live market ticks and run them through the existing UQTS-2026 pipelines (`AlphaUniverse`, `RankNet`, `MetaController`) to broadcast realistic mock data.

## 3. Dashboard Layout (The 5 Panels)
1. **The Spectral & Signal Viewer (Physicist's View)**:
   - Real-time scrolling heatmap of Morlet Wavelet CWTs (top 10 stocks).
   - Stationarity Monitor (live ADF p-values).
   - Feature SHAP Stream (live bar chart of feature importance).
2. **The Metacognition Panel (Model Health)**:
   - Bayesian Belief Gauge ($P(Valid | Result)$).
   - Adversarial Drift Alert (2D scatter plot simulating t-SNE/U-MAP of training vs live manifold).
   - Alpha Decay Curve (Cumulative information gain over time).
3. **The Cross-Sectional Ranking Grid (Quant View)**:
   - Decile Ladder (Top 10% Longs in green, Bottom 10% Shorts in red).
   - Long-Short Spread View (Cumulative performance chart).
   - Factor Exposure Heatmap (Beta, Size, Volatility).
4. **Execution & Reality Check**:
   - Implementation Shortfall (IS) Tracker (Decision vs. Execution price gap).
   - Slippage Heatmap (LOB fill locations).
   - The "Kill Switch" (Big red button to liquidate/pause).
5. **The Research Pipeline Control (Ops View)**:
   - Champion vs. Challenger backtest comparison.
   - Retraining Progress terminal log (WFO epochs and loss curves).

## 4. Implementation Steps

### Phase 1: Backend Scaffold & Data Streamer (`/cockpit_backend`)
1. Scaffold a FastAPI application.
2. Implement a `WebSocketManager` to handle frontend connections.
3. Build `streamer.py`: An `asyncio` task that utilizes `AlphaUniverse` and `MultiModalRankNet` to generate realistic sliding window data at an accelerated "live tick" rate, publishing JSON payloads to connected clients.

### Phase 2: Frontend Scaffold & Grid Layout (`/cockpit_frontend`)
1. Scaffold React using `npm create vite@latest` with Tailwind CSS.
2. Set up a dark-themed layout using CSS Grid to organize the 5 main panels in a high-density, edge-to-edge configuration.
3. Install high-performance charting libraries (e.g., Recharts or react-plotly.js).

### Phase 3: Panel Implementation & Wiring
1. Implement the React components for each of the 5 panels.
2. Establish a `useWebSocket` hook to ingest the FastAPI stream.
3. Wire the data streams to the charts (e.g., binding the CWT payload to a Heatmap component, the Bayesian Belief score to a Gauge).

## 5. Verification
- Start the FastAPI backend and React frontend.
- Verify WebSocket connection stability.
- Ensure the UI renders at 60fps without stuttering when processing continuous heatmap updates.
- Verify the "Kill Switch" correctly sends a signal back to the server to halt trading logic.