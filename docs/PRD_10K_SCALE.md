# PRD: 10,000+ Stock Scalability (DuckDB & Redis)

### Problem Statement
The current architecture, while optimized for $O(1)$ computation per tick, relies entirely on in-memory Pandas DataFrames and synchronous PyTorch inference within the FastAPI event loop. Attempting to load 10 years of data for 10,000+ stocks will result in an immediate Out-Of-Memory (OOM) crash. Furthermore, running matrix operations for 10k assets inside the web server will starve the WebSockets, creating massive latency for connected clients.

### Solution
Implement an enterprise-grade distributed architecture. We will shift historical data storage to **DuckDB** (out-of-core, vectorized execution) to handle terabytes of data without RAM limits. We will decouple the PyTorch inference into a standalone background worker that pushes pre-calculated signals to a **Redis** in-memory cache. The FastAPI web server will be relegated to a highly concurrent delivery layer, simply pulling pre-calculated payloads from Redis and streaming them to clients.

### User Stories
1.  As a quant researcher, I want to backtest and stream signals across 10,000+ equities without my laptop running out of RAM.
2.  As a user, I want the dashboard to load instantly and smoothly handle switching between thousands of tickers with <10ms latency.
3.  As a developer, I want the heavy machine learning processes to be strictly isolated from the web server, so that a model error does not disconnect connected UI clients.
4.  As a data engineer, I want historical prices stored in a persistent, queryable format on disk rather than re-downloading or re-parsing massive arrays into RAM on every boot.

### Implementation Decisions
- **Storage (DuckDB)**: Replace the Pandas `registry` in `DataEngine`. Historical data will be ingested directly into a local `data/uqts_bitemporal.ddb` file. The `get_pit_view` method will execute SQL queries to fetch only the requested slice (lookback + padding).
- **Communication (Redis)**: Introduce a Redis instance running locally (or via Docker). The system will use Redis Hashes for state storage (e.g., latest rankings) and Redis Pub/Sub for real-time tick events.
- **Worker Separation**: Create `inference_worker.py`. This script runs continuously, fetching data from DuckDB, executing `StrategyEngine`, and pushing formatted JSON payloads to Redis.
- **Streamer Refactor**: `DataStreamer` in FastAPI will no longer instantiate `StrategyEngine`. It will initialize a Redis connection pool and act as a pass-through layer, routing Redis messages to the correct WebSocket subscribers.

### Testing Decisions
Comprehensive testing is required for all new deep modules:
- **DuckDB DataEngine**: Tests must verify that `get_pit_view` returns exactly the same Point-In-Time slice as the legacy Pandas implementation, ensuring no "look-ahead" bias is introduced via SQL.
- **Redis Pub/Sub**: Tests must verify that published messages are correctly serialized, routed, and deserialized by the subscriber without data corruption.
- **Inference Worker**: Tests must verify the worker correctly handles an empty DuckDB state, processes a tick, and successfully writes the expected `GLOBAL_UPDATE` and `SPECTRAL_UPDATE` structures to Redis.

### Out of Scope
- Distributed cloud deployment (Kubernetes/Helm). This phase focuses on local scalability.
- Replacing the ML architecture (Wavelets/RankNet). The core math remains identical.

### Further Notes
This upgrade transforms UQTS-2026 from a monolithic research script into a true microservices-based high-frequency trading platform.