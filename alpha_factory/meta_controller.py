import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class BayesianMetaController:
    """
    Tracks 'Model Validity' using Bayesian Belief Updating.
    Updates the probability that the model is still capturing signal.
    """

    def __init__(self, prior_belief: float = 0.5, volatility_threshold: float = 0.05):
        self.belief = prior_belief
        self.vol_threshold = volatility_threshold

        # Performance tracking for cockpit UI
        self.correlation_history = []
        self.ic_history = []  # Information Coefficient (rolling spearman)
        self.max_history_len = 100

    def update_belief(self, realized_returns: np.ndarray, predicted_scores: np.ndarray):
        """
        Bayesian Update with Laplace Smoothing and Confidence Floor.
        Prevents permanent collapse to 0% in low-variance regimes.
        """
        if len(realized_returns) < 2:
            return self.belief

        from scipy.stats import spearmanr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Skip if no realized variance — avoids dead updates to the decay curve
            if np.std(realized_returns) < 1e-9:
                return self.belief

            correlation, _ = spearmanr(realized_returns, predicted_scores)

        if np.isnan(correlation):
            correlation = 0.0

        self.correlation_history.append(float(correlation))
        if len(self.correlation_history) > self.max_history_len:
            self.correlation_history.pop(0)

        # HORIZON MISMATCH FIX: we validate a 5-day-horizon model with 1-day
        # noisy returns. Use a 10-day rolling mean to capture the true trend
        # and dampen the sigmoid (multiplier 2.0 vs 5.0) so a single noisy
        # day doesn't crash belief.
        recent_corr = (
            np.mean(self.correlation_history[-10:]) if self.correlation_history else 0.0
        )
        likelihood_valid = 1 / (1 + np.exp(-2.0 * recent_corr))
        likelihood_invalid = 1 - likelihood_valid

        marginal = (likelihood_valid * self.belief) + (likelihood_invalid * (1 - self.belief))
        if marginal > 0:
            self.belief = (likelihood_valid * self.belief) / marginal

        self.belief = max(0.05, min(0.95, self.belief))
        return self.belief

    def get_drift_metrics(self) -> list:
        """
        Real drift telemetry: returns a 2D embedding of correlation stability
        over time. Each point is (rolling_mean, rolling_std) over a 5-sample
        window of the realized correlation history.

        A healthy model produces a tight cluster near (positive_mean, low_std).
        Drifting models spread out and walk left (mean -> 0 or negative)
        and up (std rises).

        If no history exists yet, returns an empty list (NOT synthetic data).
        Callers should treat empty results as "telemetry unavailable" and
        render accordingly.
        """
        if not self.correlation_history:
            return []

        window = 5
        hist = self.correlation_history
        points = []
        for i in range(window, len(hist) + 1):
            chunk = hist[i - window : i]
            points.append([float(np.mean(chunk)), float(np.std(chunk))])
        return points

    def get_decay_metrics(self) -> list:
        """Returns cumulative Information Gain (IC) over time."""
        if not self.correlation_history:
            return [0.0]
        return np.cumsum(self.correlation_history).tolist()

    def get_position_scaler(self):
        """Scales position size based on current belief score."""
        return self.belief
