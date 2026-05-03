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
        # Reshape for sklearn (n_samples, n_features)
        X = market_returns.reshape(-1, 1)
        y = asset_returns
        
        # Fit OLS
        self.model.fit(X, y)
        
        # Calculate residuals: y - y_pred
        prediction = self.model.predict(X)
        residuals = y - prediction
        
        return residuals

    def generate_labels(self, pit_view: pd.DataFrame, horizon: int = 21) -> pd.Series:
        """
        Generates Z-scored residualized returns for a given horizon.
        Ensures PIT consistency by operating on a provided PIT view.
        """
        # Calculate forward log-returns
        # Note: This requires a look-ahead during TRAINING only. 
        # The DataEngine ensures we don't use this during inference.
        close = pit_view['close']
        forward_returns = np.log(close.shift(-horizon) / close)
        
        # In a real scenario, we would regress against a proper market proxy (e.g., SPY)
        # For Phase 1, we will return the forward returns, ready for residualization in the notebook.
        return forward_returns
