#pragma once
#include <vector>
#include <algorithm>

namespace uqts {

struct MPCParameters {
    double market_impact_lambda = 0.01;
    double risk_aversion = 0.1;
    double alpha_decay_half_life = 10.0; // Ticks
    int horizon = 5;
};

class MPCSolver {
public:
    /**
     * Solves for the optimal trade trajectory over a planning horizon.
     * Minimizes cumulative market impact vs alpha capture.
     * Returns the target delta for the FIRST step.
     */
    static double solve_horizon(double alpha, double current_weight, const MPCParameters& params) {
        // Simplified multi-period solution for quadratic impact.
        // For a horizon H, the optimal strategy is to trade a fraction of the 
        // remaining distance to the "infinite horizon" target at each step.
        
        // Target weight if impact was zero:
        double theoretical_target = alpha / params.risk_aversion;
        
        // Trade Speed (Closed form for quadratic discrete MPC):
        // Optimal Trade = (Target - Current) * Speed
        // Where Speed = sqrt(Impact / Risk) approx for small steps.
        
        double gap = theoretical_target - current_weight;
        
        // Dynamic speed based on impact lambda:
        double speed = 1.0 / (1.0 + std::sqrt(params.market_impact_lambda * params.horizon));
        
        return gap * speed;
    }

    static double solve_single_step(double alpha, double current_weight, const MPCParameters& params) {
        return solve_horizon(alpha, current_weight, params);
    }
};

} // namespace uqts
