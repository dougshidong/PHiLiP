#ifndef __BOUND_PRESERVING_LIMITER_TESTS_H__
#define __BOUND_PRESERVING_LIMITER_TESTS_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {
/// Class used to run tests that verify implementation of bound preserving limiters
/************************************************************
* Cases include: Linear Advection (1D & 2D), Burgers' 
* Equation (1D & 2D) and Low Density Accuracy Test (2D Euler)
*************************************************************/
template <int dim, int nstate>
class BoundPreservingLimiterTests : public TestsBase
{
public:
    /// Constructor.
    explicit BoundPreservingLimiterTests(const Parameters::AllParameters* const parameters_input,
        const dealii::ParameterHandler& parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler& parameter_handler;

    /// Currently passes no matter what.
    /// Checks are included within the limiter to ensure that the principle 
    /// it is meant to preserve is satisfied at each node.
    int run_test() const override;

private:
    /// Runs full test and outputs VTK files
    int run_full_limiter_test() const;

    /// Runs convergence test and prints out results in console
    int run_convergence_test() const;

    /// Calculate and return the exact value at the point depending on the case being run
    double calculate_uexact(const dealii::Point<dim> qpoint,
        const dealii::Tensor<1, 3, double> adv_speeds,
        double final_time) const;

    /// Calculate and return the L2 Error
    std::array<double,3> calculate_l_n_error(std::shared_ptr<DGBase<dim, double>> flow_solver_dg, const int poly_degree, const double final_time) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
