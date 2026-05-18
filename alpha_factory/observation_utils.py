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
    
    EXPECTED SCALES:
    - top_10_scores: raw ranks [-0.5, 0.5] * 100.0 (i.e. -50 to 50)
    - belief: mean(top_10_scores) (i.e. -50 to 50)
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

def calculate_safe_weights(scores, concentration, asset_cap, temp=0.5, tickers=None):
    """
    Iterative weight allocator that respects asset_cap while maximizing exposure.
    Matches training logic by using raw ranks.
    """
    # SENIOR FIX (Stability): Round scores to match StrategyEngine quantization
    scores = np.round(scores, 6)
    
    # 1. Select top-K with STABLE tie-breaking
    if tickers is not None:
        # SENIOR FIX (Deterministic): Sort by (-score, ticker_name) for perfect consistency
        # across all components (Brain, General, Muscle).
        combined = []
        for i in range(len(scores)):
            combined.append({'idx': i, 'score': scores[i], 'ticker': tickers[i]})
        # Primary key: Score (desc), Secondary key: Ticker (asc)
        combined.sort(key=lambda x: (-x['score'], x['ticker']))
        top_k_indices = np.array([x['idx'] for x in combined[:concentration]])
    else:
        # Fallback to index-based stable sort if tickers not provided
        top_k_indices = np.argsort(-scores, kind='stable')[:concentration]
        
    top_scores = scores[top_k_indices]
    
    # 2. Softmax (Use 100x scaling to create the extreme peaks needed for the 350%+ baseline)
    scaled_scores = top_scores * 100.0
    exp_scores = np.exp((scaled_scores - np.max(scaled_scores)) / temp)
    weights = exp_scores / (np.sum(exp_scores) + 1e-9)
    
    # 3. Iterative Redistribution
    for _ in range(10):
        over_mask = weights > asset_cap
        under_mask = weights <= asset_cap
        
        if not np.any(over_mask):
            break
            
        overflow = np.sum(weights[over_mask] - asset_cap)
        weights[over_mask] = asset_cap
        
        if np.any(under_mask):
            under_sum = np.sum(weights[under_mask])
            if under_sum > 1e-6:
                weights[under_mask] += overflow * (weights[under_mask] / under_sum)
            else:
                weights[under_mask] += overflow / np.sum(under_mask)
        else:
            break

    # Final sanity clip and sum-to-one
    weights = np.minimum(weights, asset_cap)
    total_w = np.sum(weights)
    if total_w > 1.0:
        weights /= total_w
        
    return top_k_indices, weights
