# UQTS-2026 Version 2: The 60-Stock "Sweet Spot" Strategy

## Strategic Overview
V2 marks the transition from broad Daily alpha to deep Intraday alpha. Rather than scaling the universe wide, V2 optimizes for **Signal Resolution** and **Hardware Efficiency** by focusing on the "Super-Liquidity" leaders of the S&P 100.

## V2 Core Objectives

### 1. Universe Optimization (The 60-Stock Sweep)
*   **Concept:** Reduce the universe from 100 to 60 stocks, selecting the Top 5 leaders by market cap for each of the 12 major industrial sectors.
*   **Advantage:**
    *   **GNN Stability:** Reduces graph edges from 10,000 to 3,600, significantly cleaning up sector cross-talk noise.
    *   **VRAM Efficiency:** Frees up ~40% of GPU memory on the RTX 2070 Max-Q, enabling higher-resolution wavelet transforms.
    *   **Liquidity Integrity:** Eliminates mid-cap noise, ensuring the Alpha edge survives real-world execution slippage.

### 2. Resolution Upgrade (15-Minute Bars)
*   **Concept:** Shift from `1Day` bars to `15Min` resolution via the Alpaca SIP feed.
*   **Advantage:** Provides the LSTM and ViT with hundreds of micro-structural "frames" per simulation day, allowing the capture of intraday volatility clusters and mean-reversion algorithmic footprints.

### 3. Wavelet Texture Refinement
*   **Concept:** Use the freed VRAM to increase wavelet scale density (scales per octave).
*   **Advantage:** Provides the Vision Transformer (ViT) with much higher "texture" in the Morlet energy manifold, allowing for more precise peak/shock detection.

---
**Status:** Approved Strategic Pivot - Active Development.
