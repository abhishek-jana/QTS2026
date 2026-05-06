# UQTS-2026 Version 2: Intraday Pivot & High-Density Alpha Roadmap

## Strategic Overview
While V1 successfully captured a high-resolution daily Alpha edge (IC 0.0361) across a 100-ticker institutional universe, scaling up to 500+ stocks is computationally prohibitive (the $O(N^2)$ Pairwise Rank problem) and introduces severe slippage and noise from less liquid mid-cap equities. 

Therefore, V2 will scale **Deep** rather than **Wide**. By retaining the highly liquid S&P 100 universe and transitioning to intraday data, V2 aims to compound the Alpha edge at a significantly higher velocity.

## V2 Core Objectives

### 1. The Intraday Pivot (TimeFrame.Hour)
*   **Concept:** Shift data ingestion from `1Day` bars to `1Hour` or `15Min` bars via Alpaca's SIP feed.
*   **Advantage:** Provides the LSTM and Vision Transformer (ViT) with significantly more "frames" per day, allowing the model to learn intra-day volatility clustering, VWAP reversion, and micro-regimes.
*   **Execution:** A 2.0% IC on hourly data can be far more profitable than a 3.6% IC on daily data due to the increased frequency of compounding trades.

### 2. Wavelet Physics Optimization
*   **Concept:** The current dyadic scale structure (`[1, 2, 4, 8, 16, 32, 64, 128]`) is standard but potentially sub-optimal for capturing high-frequency intraday energy pulses.
*   **Refinement:** V2 will introduce non-dyadic, highly granular continuous wavelet scales (e.g., 10 scales per octave) to precisely isolate structural breaks and momentum shifts in the energy and tech sectors.

### 3. GNN Adjacency Refinement (Hard Sector Masks)
*   **Concept:** The `LightweightGNN` currently learns relationships using pure self-attention (every stock talks to every stock).
*   **Refinement:** Implement **Hard Sector Masking**. By injecting actual GICS sector definitions as an adjacency matrix, the GNN is restricted to learning intra-sector dynamics (e.g., Financials only heavily attend to Financials) before cross-sector dynamics. This dramatically speeds up convergence and reduces noise.

## Migration Steps for V2 Activation (Post-V1 Go-Live)
1. **Update Data Ingestor (`real_data_ingestor.py`)**: Modify the Alpaca ingestion parameters to pull `timeframe: 1Hour`.
2. **Update Feature Shapes**: Ensure the `lookback` horizon accounts for hourly bars (e.g., 63 days * 7 trading hours = 441 bars).
3. **Refine Wavelet Scales (`config.yaml`)**: Adjust `scales` array to capture the new intraday frequencies.
4. **Implement GNN Mask (`core_plugins.py`)**: Inject sector classification data into the `x_graph` transformation.

---
**Status:** Approved Roadmap - On Deck for Post-V1 Deployment.
