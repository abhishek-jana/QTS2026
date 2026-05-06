# PRD: Production-Grade Trading Infrastructure (Updated)

### Problem Statement
The initial MVP architecture was RAM-bound and synchronous, preventing scaling beyond 20 stocks. The transition required an out-of-core data strategy and asynchronous execution to support 10k+ stocks.

### Solution
- **DuckDB Integration**: Historical data is now persisted in `data/uqts_bitemporal.ddb` with SQL-driven Point-In-Time (PIT) querying, solving RAM constraints.
- **Standalone Inference Worker**: A dedicated daemon (`execution_muscle/inference_worker.py`) now runs PyTorch RankNet and Wavelet transforms, isolated from the FastAPI server.
- **Redis Pub/Sub Layer**: The system utilizes Redis for high-speed cache and asynchronous communication between the inference worker and the UI server.
- **Event-Driven OMS**: The trading system is refactored for asynchronous WebSocket reconciliation with Alpaca, ensuring reliability for partial fills and mid-day state recovery.
- **Risk Management**: A dedicated `RiskManager` daemon provides hard circuit-breaker logic, capable of emergency liquidation independent of ML signals.

### Status: IMPLEMENTED
All core architectural changes described in the original PRD have been applied. The system is currently running as a distributed architecture on the `dev` branch.

---

## 1. Institutional Data Pipeline (Polygon.io) (COMPLETE)
- Switched to `polygon-api-client` for unadjusted tick/bar ingestion into DuckDB.
- PIT logic maintained via knowledge-time-based SQL queries.

## 2. Event-Driven OMS & State Recovery (COMPLETE)
- Refactored to `asyncio` loop; WebSocket trade reconciliation integrated.
- State hydration mechanism implemented for mid-day restarts.

## 3. Hard Risk Management Circuit Breakers (COMPLETE)
- `risk_manager.py` daemon established for VaR monitoring.
- Integrated `KILL_SWITCH` for automated liquidation in extreme drawdowns.

## 4. C++ Execution Overhaul (OSQP & Covariance) (IMPLEMENTED INTERFACE)
- `mpc_solver.hpp` and `kelly_sizer.hpp` provide the C++ structures for MPC and Multivariate Kelly sizing. 
- *Note: Integration with OSQP/Eigen library remains to be compiled into the system.*

## 5. End-to-End Production Capital Test (HITL)
- Ready for staging deployment and stress testing in Alpaca Paper Environment.
