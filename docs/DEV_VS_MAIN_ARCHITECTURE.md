# Architecture Comparison: Dev vs Main

This document outlines the architectural and parameter changes made on the `dev` branch compared to the original `main` branch to achieve a high-scale, real-time performance upgrade for UQTS-2026.

| Component / Feature | Main Branch (Original) | Dev Branch (High-Scale Upgrade) | Impact / Why We Changed It |
| :--- | :--- | :--- | :--- |
| **Data Transformation** (`FractionalDifferencer`) | **Expanding Window** ($O(N^2)$). Processed the entire historical series for every tick. | **Fixed Window** ($100$ days, $O(1)$). Uses a precomputed convolution. | **Critical Performance:** Stopped the backend from slowing down over time. Prevents the "Ingestion Death Loop." |
| **Data Slicing** (`AlphaUniverse.snapshot`) | Passed all historical rows to the plugins during live inference. | Added `latest_only=True` flag. Slices data to exactly the required lookback (63) + padding (100) before processing. | **Memory/CPU:** Reduced per-tick processing from thousands of rows per asset down to ~163 rows. |
| **WebSocket Streaming** (`DataStreamer`) | **Monolithic Broadcast**. Sent heavy spectral signals for *all 20* stocks to every client every second. | **Selective Pub/Sub**. Clients subscribe to 1 ticker via `SET_TICKER`. Backend sends `SPECTRAL_UPDATE` only for that ticker. | **Bandwidth:** Reduced data payload size by ~95% per client. Prevents browser networking overload. |
| **Frontend Charting** (`App.jsx`) | Recharts / SVG lines or nothing for the live price feed. | **TradingView Lightweight Charts**. HTML5 Canvas-based Candlestick renderer. | **Rendering Speed:** SVGs crash the browser with >1,000 points. Canvas uses the GPU to render tens of thousands of candles at 60 FPS. |
| **Historical Ingestion** (`StrategyEngine`) | Blocking: Downloaded 10 years of data in the main thread during Server Startup. | **Background Task**: `asyncio.to_thread`. Server starts instantly; ingestion happens in the background. | **Availability:** The dashboard UI can load immediately instead of hanging on a "Connecting..." screen for 30 seconds. |
| **Wavelet Scales** (`SpatialPlugin`) | Overridden to `32` scales (np.geomspace). | Reverted to **`8` scales** (Dyadic: $2^1$ to $2^8$). | **Model Compatibility:** The trained TorchScript RankNet expected an input tensor of size 8, but was receiving 32, causing a `RuntimeError`. |
| **Model Modalities** (`MultiModalRankNet`) | Dual-Modality: Sequential (LSTM) and Spatial (ViT). | **Triple-Modality**: Added **Sector-Graph (GNN)** via `GraphPlugin`. | **Causality:** Fulfills the "Sector-Graph" pillar. The model now processes relational neighbor effects (e.g., supply chain cascades) alongside time-series and spectrogram data. |
| **Frontend Resilience** (`App.jsx`) | Direct object access (e.g., `data.spectral.cwt`). Crashed on partial data. | Strict Optional Chaining (`?.`) and top-level **React Error Boundary**. | **Stability:** If the backend misses a beat or a websocket packet drops, the UI falls back to "---" instead of white-screening. |

## Summary of the Upgrade

The `main` branch was built like a **batch research script**—optimized for accuracy over time, but completely unsuited for real-time live streaming. 

The `dev` branch transforms the system into a **distributed event-driven architecture**. By fixing the math to $O(1)$ constant time and decoupling the heavy data through a Pub/Sub model, the backend now runs cool and the browser stays incredibly responsive, paving the way to add thousands of new tickers to the universe.
