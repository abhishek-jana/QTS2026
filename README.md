# UQTS-2026 (Unified Quant Training System)

## Overview
UQTS-2026 is a high-performance, self-evolving Long-Short Equity ranking platform based on the **"Signal vs. Fluid"** framework. It treats market data as a non-stationary signal requiring multi-resolution analysis.

## Project Structure
- `/research_lab`: Signal discovery and alpha validation (Jupyter/Python).
- `/alpha_factory`: Industrialization and production-ready pipelines.
- `/execution_muscle`: Ultra-low latency C++26 execution wrapper.

## Key Features
- **Bi-temporal Data Engine**: Strict separation of Event Time and Knowledge Time to eliminate look-ahead bias.
- **Fractional Differentiation**: Preservation of memory while ensuring stationarity ($d \approx 0.4$).
- **Wavelet Spectrograms**: Multi-resolution analysis using Morlet wavelets on dyadic scales.
- **RankNet (LTR)**: Cross-sectional ranking of idiosyncratic alpha.

## Setup
Ensure you have `uv` installed.
```bash
uv sync
```

## Running Tests
```bash
uv run pytest
```
