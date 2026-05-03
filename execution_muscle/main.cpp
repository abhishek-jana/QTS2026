#include <iostream>
#include <vector>
#include <string>

#include "kelly_sizer.hpp"
#include "mpc_solver.hpp"

int main() {
    std::cout << "UQTS-2026: Execution Muscle Initialized (C++26)" << std::endl;
    std::cout << "Latency Target: <100us" << std::endl;
    
    // Mock signal from RankNet
    double alpha_score = 0.05;
    double variance = 0.01;
    double belief_score = 0.85;

    // 1. Calculate Kelly Fraction
    double raw_fraction = uqts::KellySizer::calculate_fraction(alpha_score, variance, belief_score);
    
    // 2. Solve MPC for optimal execution (minimal impact)
    uqts::MPCParameters params;
    double current_weight = 0.0;
    double target_weight = uqts::MPCSolver::solve_single_step(raw_fraction, current_weight, params);

    std::cout << "--- Trade Plan ---" << std::endl;
    std::cout << "Kelly Fraction: " << raw_fraction << std::endl;
    std::cout << "MPC Target Weight: " << target_weight << std::endl;

    return 0;
}
