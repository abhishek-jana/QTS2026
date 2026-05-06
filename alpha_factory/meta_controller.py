import numpy as np

class BayesianMetaController:
    """
    Tracks 'Model Validity' using Bayesian Belief Updating.
    Updates the probability that the model is still capturing signal.
    """
    def __init__(self, prior_belief: float = 0.5, volatility_threshold: float = 0.05):
        self.belief = prior_belief
        self.vol_threshold = volatility_threshold
        
        # Performance Tracking for Cockpit UI
        self.correlation_history = []
        self.ic_history = [] # Information Coefficient (rolling spearman)
        self.max_history_len = 100

    def update_belief(self, realized_returns: np.ndarray, predicted_scores: np.ndarray):
        """
        Bayesian Update with Laplace Smoothing and Confidence Floor.
        Prevents permanent collapse to 0% in low-variance regimes.
        """
        if len(realized_returns) < 2:
            return self.belief

        # 1. Calculate realized correlation
        from scipy.stats import spearmanr
        # We suppress warnings here as small universes often produce constant inputs
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Check for zero variance to avoid 'dead' updates to the Decay Curve
            if np.std(realized_returns) < 1e-9:
                return self.belief # Skip update if no price movement
                
            correlation, _ = spearmanr(realized_returns, predicted_scores)
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
            
        # Update history for metrics (only if we have variance)
        self.correlation_history.append(float(correlation))
        if len(self.correlation_history) > self.max_history_len:
            self.correlation_history.pop(0)
        
        # 2. Likelihood Function
        # Refined: Probability of seeing this correlation if the model is 'Valid'
        likelihood_valid = 1 / (1 + np.exp(-5 * (correlation - 0.01)))
        
        # 3. Bayes Rule with Learning Rate (Smoothing)
        target_belief = likelihood_valid
        alpha = 0.1
        self.belief = (1 - alpha) * self.belief + alpha * target_belief
        
        # 4. Confidence Floor
        self.belief = max(0.05, min(0.95, self.belief))
        
        return self.belief

    def get_drift_metrics(self) -> list:
        """
        Returns a simulated t-SNE latent manifold drift based on recent correlation stability.
        Format: [[x, y], ...] for Recharts scatter plot.
        """
        # We simulate the manifold drift by mapping correlation variance to a 2D space
        # In a real system, this would come from a GMM or Latent Space analysis.
        recent = self.correlation_history[-10:] if self.correlation_history else [0.0]
        variance = np.var(recent) if len(recent) > 1 else 0.05
        
        # Generate points drifting away from origin based on variance
        points = []
        for i in range(10):
            # Training cluster stays centered
            if i < 5:
                points.append([np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
            # Live cluster drifts if variance is high
            else:
                points.append([
                    np.random.normal(variance * 10, 0.2), 
                    np.random.normal(variance * 5, 0.2)
                ])
        return points

    def get_decay_metrics(self) -> list:
        """
        Returns cumulative Information Gain (IC) over time.
        """
        if not self.correlation_history:
            return [0.0]
        # Cumulative sum of correlations acts as a proxy for info gain
        return (np.cumsum(self.correlation_history)).tolist()

    def get_position_scaler(self):
        """
        Scales position size based on current belief score.
        """
        return self.belief
