# UQTS-2026 Version 3: Intraday High-Density Alpha Roadmap

## Strategic Overview
While V2 captured the "Super-Liquidity" sweet spot across a 60-ticker universe, V3 pushes the platform into high-density industrial scaling. This version targets 100+ stocks at Hourly/15-minute granularity, using complex relational graphs and high-resolution wavelet transforms.

## V3 Core Objectives

### 1. High-Density Industrial Scaling
*   **Concept:** Expand the universe back to 100+ stocks while maintaining the intraday resolution established in V2.
*   **Advantage:** Captures more cross-sectional arbitrage opportunities and rotation signals that only emerge in larger, diverse portfolios.

### 2. Relational GNN Refinement (Hard Sector Masks)
*   **Concept:** The `LightweightGNN` currently learns relationships using pure self-attention.
*   **Refinement:** Implement **Hard Sector Masking**. By injecting actual GICS sector definitions as an adjacency matrix, the GNN is restricted to learning intra-sector dynamics (e.g., Financials only heavily attend to Financials) before cross-sector dynamics. This dramatically speeds up convergence in high-node count graphs.

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

## Recommended Execution Strategy
Based on 2026 backtests, V3 should be deployed using the **Logic V3: High-Octane** regime:
- **Leverage**: 2.0x Fixed.
- **Concentration**: Top 10% of expanded universe (e.g., Top 10 for 100 stocks).
- **Risk Control**: Bayesian Belief (Metacognition) acting as an automated 0.15 threshold Kill-Switch.

---
**Status:** Approved Roadmap - On Deck for Post-V1 Deployment.
