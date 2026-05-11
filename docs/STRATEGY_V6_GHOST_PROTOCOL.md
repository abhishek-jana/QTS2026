# V6.0 "Ghost Protocol": The Skew-Aware Transformer Architecture
**Author:** Director of AI  
**Status:** DRAFT (Emergency Preparedness)  
**Objective:** Capture Leptokurtic (Fat-Tailed) Alpha in Non-Stationary Equity Markets.

---

## 1. Executive Summary: The Crisis of the Mean
Our current architecture (V5.8 RankNet) is suffering from **Feature Collapse** (IC ~0.002, Score Std ~0.001). Why? Because it was designed using standard deep learning paradigms built for Gaussian datasets (images, natural language). 

Financial markets are not Gaussian; they are **Leptokurtic**. The 2024 AI rally was defined by extreme right-tail skew (outliers like NVDA gaining 200%+ while the median stock traded sideways). Standard models minimize Mean Squared Error (MSE) across the *entire* distribution, meaning they optimize for the noise of the 99% and smooth out the 1% signal that actually generates Alpha.

To save this portfolio and capture the "Ghost Signature" of explosive outliers, we must abandon linear late-fusion and implement a **Specialized Skew-Aware Transformer (V6.0)**. This document outlines the exact architectural decisions that separate a toy model from a billion-dollar institutional brain.

---

## 2. Architectural Paradigm: Early Cross-Attention
**The Junior Mistake:** Late-Fusion (Processing Wavelets, Volume, and Momentum in separate encoders, then concatenating them into an MLP gate).
*   *Why it fails:* The MLP gate only sees the *summary* of the data. It cannot learn that "High Volume" is only important if "Wavelet Scale 4" is simultaneously peaking. 

**The Director's Solution:** Unified Early Cross-Attention.
*   We map all 32 sensors, volume matrices, and spatial wavelets into a single latent sequence. 
*   The Transformer's $Q K^T$ matrix directly computes the cross-correlation between every modality at every timestep. It builds a dynamic graph of how momentum interacts with volatility *before* attempting to rank the stocks.

---

## 3. Mission-Critical Design Choices

### A. The Attention Mechanism: Power-Scaled Sparsemax
**The Junior Mistake:** Standard `nn.Softmax(dim=-1)`.
*   *Why it fails:* Softmax is a squashing function. If NVDA has an attention score of 50 and AAPL has a score of 5, Softmax will assign them probabilities of 0.99 and 0.01. If NVDA's score jumps to 500, Softmax still outputs 0.99. It is **Blind to Magnitude** at the extremes.

**The Director's Solution:** Power-Scaled Sparsemax.
*   We replace Softmax with a parameterized, power-scaled kernel. Sparsemax forces low-relevance tokens to exactly `0.0`, routing 100% of the network's compute to the actual outliers.
*   This ensures the model ignores the "sideways" stocks and explicitly allocates attention capacity to the right-tail anomalies.

### B. Activation Functions: SwiGLU over ReLU
**The Junior Mistake:** `nn.ReLU()`.
*   *Why it fails:* ReLU enforces strict sparsity (x < 0 = 0). In financial data, a massive market gap-down creates negative tensors that instantly "kill" ReLU neurons, resulting in the exact Feature Collapse (Score Std: 0.001) we are seeing now.

**The Director's Solution:** Swish-Gated Linear Units (SwiGLU).
*   SwiGLU allows small negative gradients to flow, ensuring the network can learn from crashes without dying. It has been proven in state-of-the-art LLMs (Llama 3, Mistral) to possess superior information retention in deep layers.

### C. Normalization: RMSNorm vs. LayerNorm
**The Junior Mistake:** `nn.LayerNorm()`.
*   *Why it fails:* LayerNorm explicitly subtracts the mean of the features. If the entire market is rallying, subtracting the mean *erases the rally signature*. The model literally cannot see the bull market.

**The Director's Solution:** Root Mean Square Normalization (`RMSNorm`).
*   RMSNorm scales the variance but **does not center the mean**. This guarantees gradient stability without destroying the absolute momentum vector of the market regime.

### D. Positional Encoding: Volatility-Conditioned State-Space
**The Junior Mistake:** Sinusoidal Positional Encoding.
*   *Why it fails:* Time in the market is not linear. A week during the VIX at 12 is fundamentally different from a week during the VIX at 40. 

**The Director's Solution:** Heteroskedastic Time-Warping.
*   We scale the positional embeddings by the trailing Volatility-of-Volatility (VVIX proxy). High-volatility periods "stretch" the temporal distance between tokens, forcing the attention heads to treat crisis-days as highly distinct events rather than just "the next day."

### E. The Loss Function: Asymmetric Skew Loss
**The Junior Mistake:** Standard `MSELoss() + PairwiseRankLoss()`.
*   *Why it fails:* MSE punishes a model equally for overestimating and underestimating. In a Long-Only "Growth Hunter" mandate, we only care about the right tail.

**The Director's Solution:** Asymmetric Log-Cosh.
*   We heavily penalize the model for missing a $5\sigma$ winner (False Negative), but lightly penalize it for predicting a breakout that doesn't happen (False Positive). We want the model to take big swings. If we get the outliers right, the RL Pilot's risk-parity sizing will handle the duds.

---

## 4. Implementation Roadmap (The "Ghost" Trigger)

If the V5.8.9 Multi-Modal RankNet fails to break a sustained `0.05 IC` on the evaluation set, we immediately execute **Ghost Protocol**:

1.  **Decommission:** Delete `LightweightViT` and `LightweightGNN`.
2.  **Unify:** Flatten the `MultiModalBatch` into a single sequence `(Batch, Time, Modalities)`.
3.  **Inject:** Deploy the `SkewAwareTransformer` class utilizing `RMSNorm`, `SwiGLU`, and `Sparsemax`.
4.  **Train:** Train with Asymmetric Log-Cosh loss.

## 5. The Clinical Verification Protocol: "The Doctor"
**The Junior Mistake:** Training blindly without environmental or data-quality checks.
*   *Why it fails:* Silent OOMs, DuckDB lock contention, and signal-less datasets lead to wasted GPU hours and non-deterministic results.

**The Director's Solution:** Automated Pipeline Diagnostics (`doctor.py`).
Before every training run, the system must pass a "Full Clinical Audit" via the specialized CLI.

### A. Infrastructure Audit (`python doctor.py infra`)
*   **RAM Saturation**: Prevents OOM kills before data extraction.
*   **GPU/CUDA Check**: Ensures bit-accurate training on silicon, not emulated CPU.
*   **Zombie Reaper**: Proactively evicts stale DuckDB handles to prevent IO contention.

### B. Data Integrity Audit (`python doctor.py data`)
*   **Density Check**: Scans for bitemporal gaps in the most recent 30-day window.
*   **Skew Audit**: Snapshot of peak regimes (e.g., Jan 2024) to verify that the "Growth Hunter" labels possess the required right-tail skew for outlier detection.

### C. Model Health Audit (`python doctor.py model`)
*   **Architectural Variance**: Verifies that initial weight distributions produce non-zero standard deviation in scores (preventing instant Model Collapse).
*   **Gradient Flow**: Performs a dummy backprop to detect vanishing gradients or NaNs within the SwiGLU/Power-Scaled Attention layers before the full run.

---

*We are no longer predicting the market. We are hunting the anomalies.*