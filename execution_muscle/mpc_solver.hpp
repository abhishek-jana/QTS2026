#pragma once
#include <vector>
#include <algorithm>

namespace uqts {

struct MPCParameters {
    double market_impact_lambda = 0.01;
    double risk_aversion = 0.1;
};

class MPCSolver {
public:
    /**
     * Minimizes: -Alpha * w + Lambda * w^2 (Quadratic Market Impact)
     * Returns optimal target weight.
     */
    static double solve_single_step(double alpha, double current_weight, const MPCParameters& params) {
        // Closed form for quadratic impact + alpha
        // J = -alpha * w + 0.5 * lambda * (w - current_w)^2
        // dJ/dw = -alpha + lambda * (w - current_w) = 0
        // w = alpha / lambda + current_w
        
        double target_delta = alpha / params.market_impact_lambda;
        return current_weight + target_delta;
    }
};

} // namespace uqts
