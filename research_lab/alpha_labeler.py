import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class AlphaLabeler:
    """
    Generates target labels for the LTR model.
    Focuses on idiosyncratic alpha by residualizing returns against market proxies.
    """
    def __init__(self):
        self.model = LinearRegression()

    def residualize(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """
        Calculates the residuals of the asset returns regressed against market returns.
        Ensures the resulting alpha is uncorrelated with the market.
        """
        # Handle cases with NaN (common in shifted returns)
        mask = ~np.isnan(asset_returns) & ~np.isnan(market_returns)
        if not mask.any():
            return np.full_like(asset_returns, np.nan)
            
        X = market_returns[mask].reshape(-1, 1)
        y = asset_returns[mask]
        
        # Fit OLS
        self.model.fit(X, y)
        
        # Calculate residuals: y - y_pred
        prediction = self.model.predict(X)
        residuals = np.full_like(asset_returns, np.nan)
        residuals[mask] = y - prediction
        
        return residuals

    def residualize_universe(self, asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
        """
        Batch residualization: residualizes each column in asset_returns against market_returns.
        """
        residuals_dict = {}
        for ticker in asset_returns.columns:
            residuals_dict[ticker] = self.residualize(asset_returns[ticker].values, market_returns.values)
            
        return pd.DataFrame(residuals_dict, index=asset_returns.index)

    def apply_z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cross-sectional Z-scoring to a DataFrame of residuals.
        Calculation: (x - mean) / std across columns (tickers) for each row (time).
        """
        return df.apply(lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row - row.mean(), axis=1)

    def generate_labels(self, pit_view: pd.DataFrame, horizon: int = 21, ticker_col: str = 'ticker') -> pd.DataFrame:
        """
        Generates forward log-returns for multiple tickers.
        Returns a DataFrame indexed by event_time with tickers as columns.
        """
        # Pivot to get a wide format: rows=event_time, cols=tickers
        wide_price = pit_view.pivot(index='event_time', columns=ticker_col, values='close')
        
        # Calculate forward log-returns: ln(P_{t+N} / P_t)
        forward_returns = np.log(wide_price.shift(-horizon) / wide_price)
        
        return forward_returns
