# Architecture Upgrade: Implementation Tasks

This document tracks the vertical slices for the High-Scale Architecture Upgrade.

---

## 1. Subscription-Aware WebSocket Handshake (AFK)
### What to build
Update the FastAPI `ConnectionManager` to track specific ticker subscriptions for each connected client and modify the WebSocket endpoint to process `SET_TICKER` commands.

### Acceptance criteria
- [ ] `ConnectionManager` uses a dictionary to map `WebSocket` instances to `Optional[str]` (ticker).
- [ ] WebSocket endpoint successfully parses JSON commands with `command: "SET_TICKER"`.
- [ ] Backend logs confirm client focus updates when a command is received.

### Blocked by
None - can start immediately.

---

## 2. Decoupled Data Broadcast Logic (AFK)
### What to build
Refactor the `DataStreamer` broadcasting loop to split messages into `GLOBAL_UPDATE` (broadcast to all) and `SPECTRAL_UPDATE` (sent only to specific ticker subscribers).

### Acceptance criteria
- [ ] `GLOBAL_UPDATE` contains ladder, rankings, and metadata.
- [ ] `SPECTRAL_UPDATE` contains heavy CWT matrices and SHAP values.
- [ ] Bandwidth usage for idle clients (not viewing a specific ticker) is reduced by >80%.

### Blocked by
1. Subscription-Aware WebSocket Handshake

---

## 3. Constant-Time Math Pipeline (AFK)
### What to build
Refactor `FractionalDifferencer` to use a fixed-window approach and update `AlphaUniverse.snapshot` to perform surgical data slicing.

### Acceptance criteria
- [ ] `FractionalDifferencer` calculation time is constant ($O(1)$) regardless of total history length.
- [ ] `AlphaUniverse.snapshot` only processes the minimal required window for the current simulation tick.
- [ ] No regression in mathematical accuracy for the RankNet scores.

### Blocked by
None - can start immediately.

---

## 4. TradingView Canvas Charting (AFK)
### What to build
Integrate `lightweight-charts` into the React frontend and create a `PriceChart` component to replace/augment the current static visualizations.

### Acceptance criteria
- [ ] Main signal viewer renders 200+ candles smoothly using HTML5 Canvas.
- [ ] Chart updates in real-time as `SPECTRAL_UPDATE` messages arrive.
- [ ] UI remains responsive (no lag) during high-frequency data streams.

### Blocked by
2. Decoupled Data Broadcast Logic

---

## 5. Performance Benchmarking & Mathematical Parity (HITL)
### What to build
Final verification of the system under "load" (20+ stocks) and comparison of RankNet outputs between `main` and `dev` branches.

### Acceptance criteria
- [ ] Simulation tick time is consistently < 100ms for a 20-stock universe.
- [ ] End-to-end latency from price tick to UI update is visually imperceptible.
- [ ] Mathematically verified that RankNet scores are identical to the production baseline.

### Blocked by
3. Constant-Time Math Pipeline
4. TradingView Canvas Charting

---

## 6. V2: The 60-Stock Sweet Spot (Intraday)
### What to build
Transition the system from Daily to 15-Minute resolution with a restricted 60-stock universe and diurnal normalization.

### Acceptance criteria
- [ ] `config.yaml` universe reduced to 60 mega-cap leaders.
- [ ] `real_data_ingestor.py` fetches `15Min` bars and filters for RTH (09:30-16:00).
- [ ] Data processing pipeline implements Diurnal Standardization for returns and volume.
- [ ] Wavelet scale density increased in `config.yaml` without OOM errors on RTX 2070.

### Blocked by
5. Performance Benchmarking & Mathematical Parity
