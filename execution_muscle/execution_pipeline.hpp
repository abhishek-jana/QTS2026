#pragma once
#include <iostream>
#include <vector>

#include "kelly_sizer.hpp"
#include "mpc_solver.hpp"
#include "fractional_differencer.hpp"

namespace uqts {

/**
 * ExecutionPipeline: The high-leverage seam for production trading.
 * Orchestrates signal transformation, risk scaling, and trade optimization.
 */
class ExecutionPipeline {
public:
    ExecutionPipeline() : fd_(0.4) {}

    /**
     * Entry point for a production trade decision.
     * Takes raw signal and returns optimal portfolio weight.
     */
    double calculate_target_weight(double alpha_raw, double variance, double belief, double current_weight) {
        
        // 1. Production Kelly Scaling
        // Incorporates the Bayesian Belief Score from the Metacognition Panel.
        double kelly_fraction = KellySizer::calculate_fraction(alpha_raw, variance, belief);
        
        // 2. Multi-Period MPC Optimization
        // Minimizes market impact over a 5-tick horizon.
        MPCParameters params;
        double target_delta = MPCSolver::solve_horizon(kelly_fraction, current_weight, params);
        
        return current_weight + target_delta;
    }

    /**
     * Demonstrates the C++ Signal Path (Fractional Differentiation).
     */
    std::vector<double> process_raw_prices(const std::vector<double>& prices) {
        return fd_.transform(prices);
    }

private:
    FractionalDifferencer fd_;
};

} // namespace uqts
