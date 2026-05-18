import numpy as np
import pandas as pd
from qts_core.logger import logger

class AlphaLabeler:
    """
    Generates target labels for the Sniper strategy.
    Supports two modes:
    - 'residual': Focuses on 'Pure Alpha' by subtracting benchmark returns (Market-Neutral).
    - 'directional': Focuses on raw returns to aggressively ride Beta/Momentum.
    """
    def __init__(self, mode: str = 'directional'):
        self.mode = mode

    def generate_labels(self, pit_view: pd.DataFrame, horizon_days: int = 3, 
                        benchmark: str = 'SPY', ticker_col: str = 'ticker', 
                        timeframe: str = '15Min') -> pd.DataFrame:
        """
        Calculates forward log-returns.
        Target = ln(P_{t+3days} / P_t)
        If mode == 'residual', subtracts benchmark return.
        """
        # Ensure unique PIT entries and pivot to wide format
        df = pit_view.reset_index() if 'event_time' not in pit_view.columns else pit_view
        df = df.drop_duplicates(subset=['event_time', ticker_col])
        
        # Pivot prices: rows=time, cols=tickers
        wide_price = df.pivot(index='event_time', columns=ticker_col, values='close')
        
        # Calculate bar-based shift for the 3-day horizon
        if timeframe == '15Min':
            shift_bars = horizon_days * 26
        elif timeframe == '1Hour':
            shift_bars = horizon_days * 7
        else:
            shift_bars = horizon_days
            
        # f-returns calculation
        f_returns = np.log(wide_price.shift(-shift_bars) / wide_price)
        
        if self.mode == 'directional':
            return f_returns
            
        # Residual Mode Logic
        if benchmark in f_returns.columns:
            logger.debug(f"Residualizing against {benchmark}...")
            bench_ret = f_returns[benchmark]
            residual_returns = f_returns.sub(bench_ret, axis=0)
        else:
            logger.warning(f"Benchmark {benchmark} not found. Falling back to cross-sectional mean.")
            residual_returns = f_returns.sub(f_returns.mean(axis=1), axis=0)
            
        return residual_returns

    def apply_z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cross-sectional Z-scoring to force the model to rank 
        outperformance relative to the 60-stock universe.
        """
        return df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-9), axis=1)

    def apply_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cross-sectional ranking (0 to 1) for extreme stability.
        """
        return df.rank(axis=1, pct=True) - 0.5
