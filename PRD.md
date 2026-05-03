# UQTS-2026: Unified Quant Training System PRD

## Problem Statement
Modern quantitative trading strategies often suffer from two fatal flaws: **Look-ahead bias** (unintentional use of future information in backtests) and **Non-stationarity** (market data patterns changing over time, leading to "alpha decay"). Standard techniques like integer differentiation remove noise but destroy the "memory" of the price series, while standard backtesting environments fail to properly model the exact moment information became known to the system.

## Solution
The **UQTS-2026** platform implements a "Signal vs. Fluid" framework. It treats market data as a non-stationary signal requiring multi-resolution analysis. The solution ensures mathematical integrity via:
1.  **Bi-temporal Ingestion**: Separating *Knowledge Time* from *Event Time* to guarantee Point-in-Time (PIT) consistency.
2.  **Fractional Differentiation**: Preserving historical memory while achieving stationarity.
3.  **Wavelet Spectrograms**: Using Morlet wavelets to capture macro-regimes across dyadic scales.
4.  **Learning to Rank (LTR)**: Predicting cross-sectional Z-scores to identify the best long-short opportunities.

## User Stories
1.  As a Quant Researcher, I want to ingest market data with strict knowledge-time timestamps, so that I can be 100% certain my backtests have zero look-ahead bias.
2.  As a Signal Engineer, I want to apply fractional differentiation to a price series, so that I can maintain a stationary input for my models without losing long-range dependency information.
3.  As a Strategy Developer, I want to generate wavelet spectrograms across multiple resolutions, so that the model can distinguish between high-frequency "fluid" (noise) and low-frequency "signal" (macro-trends).
4.  As a Portfolio Manager, I want to rank stocks based on their residualized idiosyncratic alpha, so that my positions are not just proxies for common market factors or beta.
5.  As a System Architect, I want to export my research models to a high-performance C++26 environment, so that I can achieve sub-100μs inference latency in production.
6.  As a Risk Manager, I want to monitor "Model Validity" via Bayesian Belief Updating, so that the system automatically scales down positions when the alpha signal begins to decay.

## Implementation Decisions
*   **Module: DataEngine (The Bi-temporal Source)**: A deep module that wraps QuestDB/DuckDB. It exposes a single interface `get_pit_view(as_of_knowledge_time)` which asserts that no data with a `knowledge_time` > `as_of_knowledge_time` is returned.
*   **Module: AlphaCore (The Signal Processor)**: Encapsulates the `FractionalDifferencer` and `WaveletFeatureGenerator`. It takes raw bi-temporal streams and outputs normalized spectrograms.
*   **Module: AlphaRanker (The Learning Engine)**: A PyTorch-based LTR module using Pairwise RankNet loss. It operates on cross-sectional Z-scores of residualized forward returns.
*   **Module: MetaController (The Bayesian Brain)**: A monitoring layer that tracks model performance vs. expected manifold. It calculates the Bayesian Belief score used for position sizing.
*   **Module: ExecutionMuscle (The Low-Latency Wrapper)**: A C++26 project using LibTorch. It consumes ONNX/TorchScript models and executes orders via Model Predictive Control (MPC).

## Testing Decisions
*   **PIT Assertion Testing**: Every query to the `DataEngine` will be accompanied by a "time-travel" test: verify that results for $T_{backtest}$ are identical regardless of whether the database contains data up to $T_{now}$ or only up to $T_{backtest}$.
*   **Stationarity Testing**: The `FractionalDifferencer` output will be subjected to Augmented Dickey-Fuller (ADF) tests to ensure $p < 0.05$ across varying regimes.
*   **Memory Correlation Testing**: Verify that fractionally differenced series maintain a statistically significant correlation with the original series, unlike integer-differenced series (which often drop to near-zero correlation).
*   **Rank Consistency Testing**: Ensure the `AlphaRanker` produces consistent relative ordering even when input signals are subjected to Gaussian noise.

## Out of Scope
*   Real-time brokerage API integration (Phase 1 focus is Research & Pipeline).
*   Intraday tick-by-tick alpha (System is optimized for Daily/Long-term horizons).
*   Multi-asset class expansion (Initial focus is US Equities).
