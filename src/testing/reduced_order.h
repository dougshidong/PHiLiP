#ifndef __REDUCED_ORDER_H__
#define __REDUCED_ORDER_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// POD reduced order test, verifies consistency of solution
template <int dim, int nstate>
class ReducedOrder: public TestsBase
{
public:
    /// Constructor.
    ReducedOrder(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Run POD reduced order
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
