#pragma once
#include <vector>
#include <numeric>

namespace uqts {

class KellySizer {
public:
    /**
     * Kelly Criterion for normal distributions: f = mu / sigma^2
     * Scaled by Bayesian Belief Score.
     */
    static double calculate_fraction(double score, double variance, double belief) {
        if (variance <= 0) return 0.0;
        double optimal_f = score / variance;
        return optimal_f * belief;
    }
};

} // namespace uqts
