# Implementation Tasks: Production-Grade Trading Infrastructure (Updated)

This document tracks the vertical slices required to elevate the system from a high-fidelity MVP to a production-grade algorithmic trading platform capable of handling real capital.

---

## 1. Institutional Data Pipeline (Polygon.io) (DONE)
- **Status**: Implemented using `polygon-api-client` and `DuckDB`.
- **Verified**: PIT views align with legacy pandas data and successfully ingested into DuckDB.

---

## 2. Event-Driven OMS & State Recovery (DONE)
- **Status**: Refactored to `asyncio` event loop.
- **Verified**: WebSocket integration skeleton present; State recovery patterns defined for Alpaca API.

---

## 3. Hard Risk Management Circuit Breakers (DONE)
- **Status**: `risk_manager.py` daemon established for monitoring and emergency liquidation.
- **Verified**: Integrated with the system's `KILL_SWITCH` mechanism.

---

## 4. C++ Execution Overhaul (OSQP & Covariance) (INTERFACE COMPLETE)
- **Status**: Structural interfaces (`mpc_solver.hpp`, `kelly_sizer.hpp`) defined.
- **Next Step**: Link an external C++ convex optimization solver library to the build system.

---

## 5. End-to-End Production Capital Test (HITL)
- **Status**: Pending staging environment deployment.
- **Objective**: Full-stack integration test in Alpaca Paper Environment to ensure system latency, risk limits, and recovery under simulated market stress.

---

## 6. Logic V5: Institutional Guardrails (IN PROGRESS)
- [ ] **Quadratic Slippage Engine**
  - Implement 15bps (0.15%) flat commission + market impact scaling in `simulation_engine.py`.
- [ ] **Volatility Targeting Harness**
  - Refactor RL Pilot to maintain a constant volatility envelope (e.g., < 25% annualized).
- [ ] **Monte Carlo "Regime Jitter" Test**
  - Script to run 1,000 simulations with ±5% daily noise and random crash injections.
- [ ] **Liquidity Masking**
  - Use 30-day ADV (Average Daily Volume) to cap maximum position sizes at 1.5% of market depth.
