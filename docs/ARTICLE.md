# Building UQTS-2026: From Ideation to Execution

## 1. The Ideation: Signal vs. Fluid
Traditional quantitative systems fail in two key areas: **Look-ahead bias** (leaking future data into backtests) and **Non-stationarity** (models decaying as market regimes shift). UQTS-2026 was born from the "Signal vs. Fluid" philosophy. Market data is treated as a fluid, requiring multi-resolution analysis to extract the true, durable signal. 

The platform was architected as a strict 3-tier evolution: 
1. **Research Lab** (Python/Jupyter)
2. **Alpha Factory** (Automated Pipelines)
3. **Execution Muscle** (C++26 Low-Latency)

## 2. Phase 1: The Research Lab
The foundation required absolute mathematical rigor to prevent data leakage and noise fitting.
*   **Bi-temporal Ingestion:** We built a `DataEngine` that strictly separates *Event Time* (when something happened) from *Knowledge Time* (when the system learned about it). This guarantees zero look-ahead bias by hiding future price revisions during historical simulations.
*   **Fractional Differentiation:** Standard differencing destroys the long-term memory of a price series. We implemented an expanding window fractional differencer ($d \approx 0.4$) to achieve stationarity while preserving crucial long-range dependencies.
*   **Wavelet Spectrograms:** Using Morlet wavelets across dyadic scales ($2^1$ to $2^8$), we translated 1D price series into 2D market spectrograms, allowing the model to distinguish between high-frequency noise and low-frequency macro-regimes.
*   **Residualized Alpha Labeling:** To prevent the model from simply proxying market beta, the `AlphaLabeler` residualizes forward returns against a market proxy (e.g., SPY) and applies cross-sectional Z-scoring to isolate pure idiosyncratic alpha.

## 3. Phase 2: The Alpha Factory
With the signal physics verified, we industrialized the training process.
*   **RankNet LTR:** We implemented a Learning-to-Rank (LTR) neural network in PyTorch. Using a Pairwise RankNet loss, the model learns the relative cross-sectional ordering of the asset universe rather than absolute price predictions.
*   **Multi-Modal Fusion:** The architecture evolved to a two-stream model: an LSTM for temporal sequences and a custom Vision Transformer (ViT) for spatial wavelet spectrograms, fused before the ranking head.
*   **Walk-Forward Optimization (WFO):** A dynamic `WFOEngine` was built to step chronologically through history, retraining the model continuously while strictly honoring the Point-in-Time boundaries set by the `DataEngine`.

## 4. Phase 3: The Execution Muscle
Research alpha is theoretical until executed efficiently.
*   **TorchScript Bridge:** The Multi-Modal model is serialized via TorchScript Tracing, allowing the exact computational graph to be loaded and executed natively in a high-performance C++ environment.
*   **C++26 Execution:** The final layer is written in modern C++26 to achieve sub-100μs latency.
*   **MPC & Kelly Sizing:** The execution engine utilizes a Model Predictive Control (MPC) solver to navigate trades with minimal market impact. Final position sizes are scaled using the Kelly Criterion, strategically weighted by the Bayesian belief score.

## 5. The Execution Strategy: Tracer Bullets & TDD
The project was executed methodically using vertical "tracer bullet" slices tracked via GitHub issues. Every slice delivered a verifiable, end-to-end piece of functionality. Test-Driven Development (TDD) was enforced at every step, ensuring that critical behaviors—like knowledge-time isolation and cross-sectional centering—were mathematically proven before any complex modeling began. 

The result is UQTS-2026: a mathematically sound, high-performance, self-evolving Long-Short Equity ranking platform.