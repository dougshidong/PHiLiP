#ifndef __RRK_NUMERICAL_ENTROPY_CONSERVATION_CHECK__
#define __RRK_NUMERICAL_ENTROPY_CONSERVATION_CHECK__

#include <deal.II/base/convergence_table.h>

#include "dg/dg_base.hpp"
#include "tests.h"
#include "flow_solver/flow_solver.h"

namespace PHiLiP {
namespace Tests {

/// Verify numerical_entropy conservation for inviscid Burgers using split form and RRK
template <int dim, int nstate>
class RRKNumericalEntropyConservationCheck: public TestsBase
{
public:
    /// Constructor
    RRKNumericalEntropyConservationCheck(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:

    /// Reinitialize parameters. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params(bool use_rrk, double time_step_size) const;
    
    /// Compare the numerical_entropy after flow simulation to initial, and return test fail int
    int compare_numerical_entropy_to_initial(
            const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
            const double initial_numerical_entropy,
            const double final_time_actual,
            bool expect_conservation
            ) const;

    /// runs flow solver. Returns 0 (pass) or 1 (fail) based on numerical_entropy conservation of calculation.
    int get_numerical_entropy_and_compare_to_initial(
            const Parameters::AllParameters params,
            const double numerical_entropy_initial,
            bool expect_conservation
            ) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
