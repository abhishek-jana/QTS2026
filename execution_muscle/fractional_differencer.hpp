#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace uqts {

/**
 * High-performance Fractional Differencer (C++ implementation).
 * Maintains an expanding window or fixed-window kernel for stationarity.
 */
class FractionalDifferencer {
public:
    FractionalDifferencer(double d, double threshold = 1e-4) : d_(d), threshold_(threshold) {
        generate_weights(252); // Precompute for common year window
    }

    /**
     * Transforms a raw price vector into a stationary series.
     */
    std::vector<double> transform(const std::vector<double>& input) {
        size_t n = input.size();
        std::vector<double> output;
        output.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            double val = 0.0;
            // Apply weights up to current history
            size_t window_size = std::min(i + 1, weights_.size());
            for (size_t k = 0; k < window_size; ++k) {
                val += weights_[k] * input[i - k];
            }
            output.push_back(val);
        }
        return output;
    }

private:
    void generate_weights(size_t size) {
        weights_.clear();
        weights_.push_back(1.0);
        for (size_t k = 1; k < size; ++k) {
            double w_k = -weights_.back() * (d_ - k + 1.0) / k;
            if (std::abs(w_k) < threshold_ && k > 63) break; // Truncate if below threshold
            weights_.push_back(w_k);
        }
    }

    double d_;
    double threshold_;
    std::vector<double> weights_;
};

} // namespace uqts
