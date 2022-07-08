#ifndef __FINITE_DIFFERENCE_SENSITIVITY__
#define __FINITE_DIFFERENCE_SENSITIVITY__

#include "tests.h"
#include "dg/dg.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver/flow_solver_factory.h"
#include <fstream>

namespace PHiLiP {
namespace Tests {

/// Burgers Rewienski sensitivity
template <int dim, int nstate>
class FiniteDifferenceSensitivity: public TestsBase
{
public:
    /// Constructor.
    FiniteDifferenceSensitivity(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Run test
    int run_test () const override;

    /// Renitialize parameters, necessary because parameters created for the test are constant
    Parameters::AllParameters reinit_params(double perturbation) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
