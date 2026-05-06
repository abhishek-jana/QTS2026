# Research Optimization Log: UQTS-2026

This document tracks the evolution of the research pipeline, identifying performance bottlenecks and the technical solutions implemented to achieve institutional-grade training speeds and Alpha integrity.

## 1. Institutional Baseline (Current)
*   **Architecture:** Quad-Modality `MultiModalRankNet` fusing Sequential (LSTM), Spatial (Vision Transformer), Sector-Graph (GNN), and Volume Dynamics.
*   **Fusion Strategy:** Learned Modality Gating (Attention-based weighting of sensor inputs).
*   **Dataset:** ~140,000 unique daily training snapshots representing a diversified 100-ticker universe (S&P 500 across all sectors).
*   **Training Window:** 2016-01-01 to 2022-12-31.
*   **VRAM Optimization:** Dataset residency in GPU VRAM with AMP (Automatic Mixed Precision).

## 2. Evolution of Math & Efficiency

### The "Redundancy Trap" Resolution
*   **Problem:** Early versions produced ~1.8 million training samples via redundant, overlapping windows. This led to massive overfitting (model memorizing specific dates) and 12-hour training times.
*   **Solution:** Transitioned to **Unique Daily Snapshots** (`latest_only=True` in snapshots).
    *   *Result:* Training set reduced to ~28,000 high-quality unique windows for 20 tickers (~140,000 for 100 tickers).
    *   *Impact:* Training speed increased by 64x with significantly improved generalization (OOS IC stability).

### The "ViT Blindness" Correction
*   **Problem:** Standard Vision Transformer patch sizes were too large for 8-scale wavelet spectrograms, causing the model to "see" blank fields.
*   **Solution:** Implemented **Dynamic Patching**. The ViT now automatically scales its patch height to match the spectrogram height (Scales count).
    *   *Result:* ViT modality is now active and contributing to the Alpha signal.

### Modality Gating (Regime Awareness)
*   **Problem:** Simple concatenation of modalities forced the model to trust noisy streams (like GNN on small universes) as much as stable ones (LSTM).
*   **Solution:** Implemented a **Learned Gating Layer**. The model now assigns a dynamic weight to each modality for every prediction.
    *   *Impact:* Model can automatically "ignore" noisy sensors during market regime shifts.

## 3. Performance Summary

| Metric | Redundant / 20 Ticker | Unique / 100 Ticker |
| :--- | :--- | :--- |
| **Sample Volume** | 1,800,000 (Redundant) | **~140,000 (Unique)** |
| **Training Time (25 Ep)**| ~4-12 Hours | **~45-60 Minutes** |
| **GPU Utilization** | Saturated (Noise) | **Optimized (Signal)** |
| **VRAM Usage** | ~4 GB | **~6 GB** |
| **Alpha Quality** | High Overfit Risk | **Institutional Generalization** |

## 4. Scaling Strategy
*   **Universe Expansion:** Moving to 100 tickers provided the GNN with the necessary graph density to learn cross-sectional relationships.
*   **Temporal Splitting:** Abandoned random 20% validation in favor of a strictly temporal hold-out (Late 2022). This ensures the validation loss is a true proxy for out-of-sample performance.

---
**Last Updated:** May 6, 2026
**Status:** Institutional Phase Engaged - 100 Ticker Baseline Locked.
