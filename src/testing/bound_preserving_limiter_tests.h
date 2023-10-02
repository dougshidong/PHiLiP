#ifndef __BOUND_PRESERVING_LIMITER_TESTS_H__
#define __BOUND_PRESERVING_LIMITER_TESTS_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/************************************************************
* Class used to run full tests (with output) as well as
* convergence tests for bound_preserving_limiter cases.
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
    int run_test() const override;

private:
    int run_full_limiter_test() const;

    int run_convergence_test() const;

    double calculate_uexact(const dealii::Point<dim> qpoint,
        const dealii::Tensor<1, 3, double> adv_speeds,
        double final_time) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
