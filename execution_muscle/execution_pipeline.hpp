#pragma once
#include "kelly_sizer.hpp"
#include "mpc_solver.hpp"

namespace uqts {

/**
 * ExecutionPipeline: A Deep Module encapsulating sizing and execution logic.
 * This provides a 'Seam' for the system where execution strategy can be 
 * swapped or modified without affecting the caller.
 */
class ExecutionPipeline {
public:
    ExecutionPipeline() = default;
    
    /**
     * Initializes the pipeline with specific MPC parameters.
     */
    explicit ExecutionPipeline(const MPCParameters& params) : m_params(params) {}

    /**
     * Executes the full execution sequence:
     * 1. Kelly Sizing based on alpha, variance, and belief.
     * 2. MPC Solver to determine the optimal target weight given market impact lambda.
     */
    double calculate_target_weight(double alpha, double variance, double belief, double current_weight) {
        double raw_fraction = KellySizer::calculate_fraction(alpha, variance, belief);
        return MPCSolver::solve_single_step(raw_fraction, current_weight, m_params);
    }

    // Accessors for parameter management
    void set_parameters(const MPCParameters& params) { m_params = params; }
    const MPCParameters& get_parameters() const { return m_params; }

private:
    MPCParameters m_params;
};

} // namespace uqts
