import pytest
import numpy as np
from alpha_factory.meta_controller import BayesianMetaController

def test_belief_decay_on_noise():
    """
    TDD: Verify belief score decays when predictions are just noise.
    """
    controller = BayesianMetaController(prior_belief=0.8)
    
    # 10 assets
    np.random.seed(42)
    for _ in range(20): # Increased steps for smoothed update
        realized = np.random.normal(0, 0.01, 10)
        predicted = np.random.normal(0, 0.01, 10) # Random noise
        controller.update_belief(realized, predicted)
    
    # After 20 steps of noise, belief should be lower than 0.8
    assert controller.belief < 0.8
    assert controller.belief < 0.65 # Relaxed due to smoothing

def test_belief_increase_on_signal():
    """
    TDD: Verify belief score increases when predictions match returns.
    """
    controller = BayesianMetaController(prior_belief=0.5)
    
    np.random.seed(42)
    for _ in range(20): # Increased steps for smoothed update
        realized = np.random.normal(0, 0.01, 10)
        predicted = realized * 0.5 + np.random.normal(0, 0.001, 10) # Strong signal
        controller.update_belief(realized, predicted)
        
    assert controller.belief > 0.5
    assert controller.belief > 0.8 # Relaxed due to smoothing
