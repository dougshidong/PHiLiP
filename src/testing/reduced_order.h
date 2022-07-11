#ifndef __REDUCED_ORDER_H__
#define __REDUCED_ORDER_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
#include "pod_adaptive_sampling.h"
#include "functional/functional.h"
#include "functional/lift_drag.hpp"
#include "parameters/all_parameters.h"
#include <deal.II/numerics/solution_transfer.h>
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/numbers.h>
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include <iostream>
#include "functional/functional.h"

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
