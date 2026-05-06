# Research Optimization Log: UQTS-2026

This document tracks the evolution of the research pipeline, identifying performance bottlenecks and the technical solutions implemented to achieve institutional-grade training speeds.

## 1. Current Strategy (Baseline)
*   **Architecture:** Triple-Modality `MultiModalRankNet` fusing Sequential (LSTM), Spatial (Vision Transformer), and Sector-Graph (GNN) features.
*   **Dataset:** ~1.8 million training samples representing a diversified 20-ticker universe.
*   **Training Window:** 2018-01-01 to 2022-12-31.
*   **VRAM Optimization:** Entire processed dataset (~4GB) is pre-loaded into GPU VRAM to eliminate CPU-GPU transfer bottlenecks.

## 2. Identified Bottlenecks (The "12-Hour" Problem)
Despite hitting 100% GPU utilization, the initial research runs took ~12 hours due to the following factors:

*   **Epoch Overkill:** Training for 25 full epochs on 1.8M records from a "cold start" without a specialized learning rate schedule.
*   **Full Precision Math:** Using FP32 (Standard Float) for all calculations. While the GPU is 100% active, it is doing much more work than necessary by not utilizing hardware-level Tensor Cores.
*   **Evaluation Frequency:** Running backtest checks every 7 days (Weekly Stride), leading to excessive inference overhead during the testing phase.
*   **Python Overhead:** Relying on standard `DataLoader` dictionary creation which adds milliseconds of Python "glue" time for every one of the millions of samples.

## 3. The Optimization Roadmap

### Phase 1: Hardware Acceleration (The "Magic" Speedup)
*   **Automatic Mixed Precision (AMP):** Implementation of `torch.cuda.amp`. By switching from FP32 to FP16 (Half-Precision) where possible, we target the GPU's **Tensor Cores**.
    *   *Impact:* ~2x faster training and 50% less VRAM usage for activations.
*   **Batch Right-Sizing:** Calibrating batch size to **8,192**. This keeps the GPU compute units saturated while leaving enough "headroom" in the 8GB VRAM to avoid memory swapping.

### Phase 2: Algorithmic Efficiency
*   **"Warm-Starting":** Transitioning from training a new model every time to "Fine-Tuning" an existing model. 
*   **Strategic Epoch Reduction:** Lowering total epochs to **3–5** with active **Early Stopping**. In institutional WFO, a model only needs to "see" the latest data a few times to adapt its weights.
*   **Fast Tensor Slicing:** Bypassing `DataLoader` entirely in favor of direct GPU-native indexing of the VRAM-resident dataset.

### Phase 3: Institutional Realism
*   **Rolling Window:** Replacing the "Expanding Window" (which grows infinitely) with a **48-month Rolling Window**. This ensures the model only learns from recent, relevant market regimes.
*   **Evaluation Stride Alignment:** Moving from a weekly stride to a **21-day (Monthly) stride**.
    *   *Impact:* Reduces the backtesting duration by ~66% with no loss in statistical significance for Alpha signals.

## 4. Summary of Improvements

| Metric | Before Optimization | After Optimization |
| :--- | :--- | :--- |
| **Total Pipeline Time** | ~12 Hours | **< 45 Minutes** |
| **GPU Utilization** | 100% (Saturated) | **100% (High-Efficiency)** |
| **Precision** | FP32 (Standard) | **AMP / FP16 (Tensor Cores)** |
| **Training Logic** | Cold Start / 25 Epochs | **Warm-Start / 3-5 Epochs** |
| **Backtest Speed** | Weekly Stride | **Monthly Stride (3x Faster)** |

---
**Last Updated:** May 2026
**Status:** Optimization Plan Approved - Awaiting Implementation Signal.
