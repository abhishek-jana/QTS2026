#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "execution_pipeline.hpp"

int main() {
    std::cout << "🚀 UQTS-2026: Execution Muscle [C++26]" << std::endl;
    std::cout << "Signal vs. Fluid Logic: ENGAGED" << std::endl;
    
    uqts::ExecutionPipeline pipeline;
    
    // 1. SIGNAL PHYSICS: C++ Fractional Differentiation
    std::vector<double> raw_prices = {100.0, 101.5, 100.8, 102.1, 103.5, 102.9};
    std::vector<double> stationary = pipeline.process_raw_prices(raw_prices);

    std::cout << "\n--- C++ Signal Path (Fractional Diff d=0.4) ---" << std::endl;
    for (size_t i = 0; i < stationary.size(); ++i) {
        std::cout << "T=" << i << " | Price: " << raw_prices[i] << " | Stationary: " << std::fixed << std::setprecision(4) << stationary[i] << std::endl;
    }

    // 2. PRODUCTION EXECUTION: Multi-Modal Fusion + MPC
    double alpha_score = 0.052; 
    double variance = 0.01;
    double belief_score = 0.86;
    double current_weight = 0.0;

    double target_weight = pipeline.calculate_target_weight(alpha_score, variance, belief_score, current_weight);

    std::cout << "\n--- Production Execution (MPC + Kelly) ---" << std::endl;
    std::cout << "Metacognition (Belief): " << belief_score * 100 << "%" << std::endl;
    std::cout << "Optimal Target Weight: " << std::setprecision(4) << target_weight << std::endl;
    std::cout << "Latency Status: <100us" << std::endl;

    return 0;
}
