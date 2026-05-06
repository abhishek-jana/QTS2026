# PRD: High-Scale Financial Dashboard Architecture Upgrade

### Problem Statement
The initial UQTS-2026 dashboard architecture suffered from severe performance bottlenecks. Specifically, the backend was re-processing entire historical datasets ($O(N^2)$ complexity) every 500ms and broadcasting heavy wavelet spectrogram data for every stock in the universe to every connected client. This resulted in high server CPU usage, massive bandwidth consumption, and browser-side rendering lag (SVG-based).

### Solution
Transition to a subscription-aware, constant-time architecture. This involves:
1.  **Selective Subscriptions**: Clients only receive heavy spectral data for the specific ticker they are focused on.
2.  **Constant-Time Math**: Implementation of fixed-window fractional differentiation and surgical data slicing to ensure $O(1)$ performance per simulation tick.
3.  **Canvas Rendering**: Swapping SVG-based charts for TradingView's Lightweight Charts (HTML5 Canvas) to offload rendering to the GPU.

### User Stories
1.  As a quant analyst, I want the dashboard to load instantly and remain responsive even with 20+ stocks in the universe.
2.  As a trader, I want to click any stock in the ranking ladder and immediately see its live price feed and wavelet spectrogram.
3.  As a developer, I want the backend's resource usage to remain flat over time, regardless of the simulation's duration.
4.  As a remote user, I want the system to be bandwidth-efficient by only streaming detailed data for the asset I am actively inspecting.

### Implementation Decisions
- **Selective Data Streaming**: The WebSocket protocol is updated to support a `SET_TICKER` command. The backend now maintains a per-client subscription state.
- **Message Decoupling**: Payloads are split into `GLOBAL_UPDATE` (rankings/ladder for all) and `SPECTRAL_UPDATE` (heavy signals for subscribers).
- **Optimized Math Pipeline**: 
    - `FractionalDifferencer` uses a fixed 100-day window.
    - `AlphaUniverse.snapshot` employs surgical slicing to only process the minimal required lookback window.
- **GPU-Accelerated Charts**: Integrated `lightweight-charts` for candle rendering, providing 60FPS performance.

### Testing Decisions
- **Mathematical Integrity**: Unit tests will verify that the fixed-window `FractionalDifferencer` produces results consistent with the original expanding-window implementation.
- **Subscription Reliability**: Integration tests will confirm that clients only receive spectral data for their subscribed tickers and that switching tickers correctly updates the stream.
- **Latency Benchmarking**: Monitor the simulation tick time to ensure it remains below the 500ms threshold for a 20-stock universe.

### Out of Scope
- Distributed caching with Redis (planned for next phase).
- Multi-user authentication.
- Historical data export features.

### Further Notes
This upgrade significantly reduces the "Death Loop" risk identified during initial testing and provides a professional-grade foundation for scaling to hundreds of tickers.
