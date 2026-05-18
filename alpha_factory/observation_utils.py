import numpy as np
from qts_core.logger import logger

def build_rl_observation(
    top_10_scores, 
    bot_10_scores, 
    belief, 
    drawdown, 
    vol_21, 
    current_lev, 
    vol_vel, 
    spy_trend, 
    rsi, 
    spy_ret, 
    cash_ratio, 
    dow
):
    """
    Unified 32-sensor observation builder.
    Ensures training and deployment see the same distribution.
    """
    obs = np.concatenate([
        top_10_scores, 
        bot_10_scores,
        [belief, drawdown, vol_21, current_lev],
        [vol_vel, spy_trend, rsi, spy_ret],
        [cash_ratio, 0.0, 1.0, dow],
    ]).astype(np.float32)
    
    # Standardize scaling and handle NaNs
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(obs, -10.0, 10.0)

def calculate_safe_weights(scores, concentration, asset_cap, temp=0.5):
    """
    Allocation weights that STRICTLY enforce asset_cap.
    If concentration * asset_cap < 1.0, the remaining weight stays as cash.
    """
    top_k_indices = np.argsort(scores)[-concentration:][::-1]
    top_scores = scores[top_k_indices]
    
    # Scale scores for softmax stability
    scaled_scores = top_scores * 100.0
    exp_scores = np.exp((scaled_scores - np.max(scaled_scores)) / temp)
    raw_weights = exp_scores / (np.sum(exp_scores) + 1e-9)
    
    # SENIOR FIX (Risk): Strictly enforce asset_cap without violating it during renormalization.
    # If the cap is 0.15 and concentration is 5, max exposure is 0.75. 
    # Renormalizing to 1.0 would push weights to 0.20 (VIOLATION).
    
    weights = np.minimum(raw_weights, asset_cap)
    
    # We only renormalize if the sum exceeds 1.0 (rare if asset_cap is low)
    # or if we want to redistribute the "shaved off" weight from the cap.
    # BUT we must never exceed asset_cap.
    
    total_w = np.sum(weights)
    if total_w > 1.0:
        weights = (weights / total_w) * 1.0
        # Second pass to ensure no single asset exceeds cap after down-scaling
        weights = np.minimum(weights, asset_cap)
        
    return top_k_indices, weights
