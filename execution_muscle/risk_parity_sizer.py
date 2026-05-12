import pandas as pd
import numpy as np
from typing import Dict, List
from qts_core.logger import logger

class RiskParitySizer:
    """
    Sniper V7.0 Deterministic Allocation Engine.
    Allocates capital inversely proportional to trailing 20-day volatility.
    Ensures Equal Risk Contribution across the portfolio.
    """
    def __init__(self, data_engine, lookback_days: int = 20, max_weight: float = 0.15):
        self.data_engine = data_engine
        self.lookback = lookback_days
        self.max_weight = max_weight

    def get_target_weights(self, tickers: List[str], as_of: pd.Timestamp) -> Dict[str, float]:
        """
        Calculates Inverse-Volatility weights for the provided tickers.
        """
        volatilities = {}
        for ticker in tickers:
            try:
                # We need enough historical data to calculate lookback returns
                start_time = as_of - pd.Timedelta(days=self.lookback + 10)
                view = self.data_engine.get_pit_view(ticker, as_of)
                if view.empty: continue
                
                # Resample intraday to daily to get standard daily volatility
                daily_closes = view['close'].resample('1D').last().dropna()
                if len(daily_closes) < 5: continue
                
                daily_rets = daily_closes.pct_change().dropna()
                # Annualized Volatility (assuming 252 trading days)
                vol = daily_rets.tail(self.lookback).std() * np.sqrt(252)
                
                if vol > 1e-6:
                    volatilities[ticker] = vol
                else:
                    volatilities[ticker] = np.inf
            except Exception as e:
                logger.debug(f"RiskParitySizer: Failed to calc vol for {ticker}: {e}")
                
        if not volatilities:
            # Fallback to equal weight if vol calc fails
            return {t: 1.0/len(tickers) for t in tickers}
            
        inv_vols = {t: 1.0 / v for t, v in volatilities.items() if v < np.inf}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {}
        for t in tickers:
            if t in inv_vols and total_inv_vol > 0:
                raw_w = inv_vols[t] / total_inv_vol
                weights[t] = min(raw_w, self.max_weight)
            else:
                weights[t] = 0.0
                
        # Re-normalize after clipping max weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w for t, w in weights.items()}
            
        return weights
