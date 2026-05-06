#ifndef MPC_SOLVER_HPP
#define MPC_SOLVER_HPP

#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>

/**
 * Model Predictive Control (MPC) Solver.
 * Balances Alpha Decay vs Market Impact using Convex Optimization.
 */
class MPCSolver {
public:
    MPCSolver(double lambda_impact, double risk_aversion, int horizon) 
        : lambda_(lambda_impact), rho_(risk_aversion), h_(horizon) {}

    /**
     * Solves for optimal trade trajectory (v_t)
     * Minimize: Sum_{t=1 to H} [ lambda * v_t^2 - alpha_t * v_t ]
     * Subject to: Sum(v_t) = total_qty
     */
    std::vector<double> optimize_trade_trajectory(
        const std::vector<double>& alpha_decay, 
        double total_qty
    ) {
        std::vector<double> trajectory;
        int steps = std::min((int)alpha_decay.size(), h_);
        
        // Closed-form solution for quadratic impact penalty:
        // v_t = (alpha_t + nu) / (2 * lambda)
        // where nu is the lagrange multiplier for the sum constraint.
        
        double alpha_sum = std::accumulate(alpha_decay.begin(), alpha_decay.begin() + steps, 0.0);
        double nu = (2.0 * lambda_ * total_qty - alpha_sum) / steps;

        for (int t = 0; t < steps; ++t) {
            double v_t = (alpha_decay[t] + nu) / (2.0 * lambda_);
            trajectory.push_back(v_t);
        }

        std::cout << "MPC Solver: Optimized trajectory for qty " << total_qty 
                  << " over " << steps << " steps." << std::endl;
        return trajectory;
    }

private:
    double lambda_; // Market impact penalty
    double rho_;    // Risk aversion (variance penalty)
    int h_;         // Planning horizon
};

#endif
