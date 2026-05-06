#ifndef KELLY_SIZER_HPP
#define KELLY_SIZER_HPP

#include <vector>
#include <cmath>

/**
 * Multivariate Kelly Sizer.
 * Incorporates Asset Covariance Matrix to optimize position sizing.
 */
class KellySizer {
public:
    KellySizer() {}

    // Calculates position sizes using Multivariate Kelly Criterion
    std::vector<double> calculate_sizes(
        const std::vector<double>& scores, 
        const std::vector<std::vector<double>>& covariance_matrix
    ) {
        // Placeholder for matrix inversion logic: f = Cov^-1 * mu
        std::vector<double> sizes(scores.size(), 0.01); // Default 1%
        return sizes;
    }
};

#endif
