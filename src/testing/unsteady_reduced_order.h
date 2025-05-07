#ifndef __UNSTEADY_REDUCED_ORDER_H__
#define __UNSTEADY_REDUCED_ORDER_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Unsteady POD reduced order test, verifies consistency of solution and implementation of threshold function
template <int dim, int nstate>
class UnsteadyReducedOrder: public TestsBase
{
public:
    /// Constructor.
    UnsteadyReducedOrder(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Run Unsteady POD reduced order
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
