import numpy as np

class BayesianMetaController:
    """
    Tracks 'Model Validity' using Bayesian Belief Updating.
    Updates the probability that the model is still capturing signal.
    """
    def __init__(self, prior_belief: float = 0.5, volatility_threshold: float = 0.05):
        self.belief = prior_belief
        self.vol_threshold = volatility_threshold

    def update_belief(self, realized_returns: np.ndarray, predicted_scores: np.ndarray):
        """
        Bayesian Update: P(Valid|Result) = P(Result|Valid) * P(Valid) / P(Result)
        Likelihood P(Result|Valid) is based on the Spearman correlation between 
        predicted ranks and realized returns.
        """
        # 1. Calculate realized correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(realized_returns, predicted_scores)
        
        # 2. Likelihood Function
        # If valid, expect high correlation (e.g., > 0.05).
        # We use a simple sigmoid to map correlation to likelihood.
        likelihood_valid = 1 / (1 + np.exp(-10 * (correlation - 0.02)))
        likelihood_invalid = 1 - likelihood_valid
        
        # 3. Bayes Rule
        numerator = likelihood_valid * self.belief
        denominator = (likelihood_valid * self.belief) + (likelihood_invalid * (1 - self.belief))
        
        self.belief = numerator / denominator if denominator > 0 else 0.0
        return self.belief

    def get_position_scaler(self):
        """
        Scales position size based on current belief score.
        Simple linear scaling for now.
        """
        return self.belief
