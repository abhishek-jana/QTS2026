# Implementation Tasks: 10,000+ Stock Scalability

This document tracks the vertical slices for integrating DuckDB and Redis to achieve 10k+ stock scalability.

---

## 1. DuckDB Storage Engine Integration (AFK)
### What to build
Rewrite `DataEngine` in `research_lab/data_engine.py` to persist historical data to a local `data/uqts_bitemporal.ddb` file instead of an in-memory Pandas DataFrame. Update `get_pit_view` to use vectorized SQL queries to fetch the required `lookback` + `padding` slice.

### Acceptance criteria
- [ ] `data/uqts_bitemporal.ddb` file is created upon ingestion.
- [ ] Memory profile remains flat during ingestion of a large universe.
- [ ] `get_pit_view` returns identical results to the legacy Pandas implementation.
- [ ] Unit tests for `DataEngine` pass.

### Blocked by
None - can start immediately.

---

## 2. Standalone Inference Worker (AFK)
### What to build
Create `execution_muscle/inference_worker.py` to run the `StrategyEngine` (including `WaveletFeatureGenerator` and `RankNet`) in an isolated, continuous loop, pulling data from DuckDB.

### Acceptance criteria
- [ ] Worker runs independently of the FastAPI server.
- [ ] Worker successfully polls DuckDB and executes the math pipeline without throwing errors.
- [ ] Logs confirm simulation ticks are processing.

### Blocked by
1. DuckDB Storage Engine Integration

---

## 3. Redis Pub/Sub Caching Layer (AFK)
### What to build
Integrate a Redis client into the `inference_worker.py` to push the serialized `GLOBAL_UPDATE` (rankings, metacognition) and `SPECTRAL_UPDATE` (cwt, shap) payloads into Redis channels.

### Acceptance criteria
- [ ] Worker successfully connects to a local Redis instance.
- [ ] `GLOBAL_UPDATE` payloads are published to a global channel.
- [ ] `SPECTRAL_UPDATE` payloads are published to specific ticker channels (e.g., `stream:NVDA`).

### Blocked by
2. Standalone Inference Worker

---

## 4. FastAPI Redis Pass-Through (AFK)
### What to build
Refactor `cockpit_backend/streamer.py` and `main.py` to remove all PyTorch and `StrategyEngine` imports. The streamer should subscribe to the Redis channels and route messages to the connected WebSockets based on their active subscriptions.

### Acceptance criteria
- [ ] FastAPI starts instantly without downloading data or loading ML models.
- [ ] UI clients receive data via WebSockets identical to the previous monolithic architecture.
- [ ] Switching tickers dynamically subscribes the client to the correct Redis channel.

### Blocked by
3. Redis Pub/Sub Caching Layer

---

## 5. 10k Scale Test & Benchmarking (HITL)
### What to build
Final human-in-the-loop verification. Synthesize 10,000+ tickers into DuckDB and run the full distributed stack to verify latency and memory usage.

### Acceptance criteria
- [ ] DuckDB database handles 10k+ tickers without causing an OOM error.
- [ ] FastAPI memory footprint remains negligible (<100MB).
- [ ] UI remains responsive (<10ms UI lag) when switching focus between tickers in a massive universe.

### Blocked by
4. FastAPI Redis Pass-Through