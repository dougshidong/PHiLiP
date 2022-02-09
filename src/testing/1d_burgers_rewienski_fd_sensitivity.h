#ifndef __1D_BURGERS_REWIENSKI_FD_SENSITIVITY__
#define __1D_BURGERS_REWIENSKI_FD_SENSITIVITY__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
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

namespace PHiLiP {
namespace Tests {

/// Burgers Rewienski sensitivity
template <int dim, int nstate>
class BurgersRewienskiSensitivity: public TestsBase
{
public:
    /// Constructor.
    BurgersRewienskiSensitivity(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;

    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> initialize_ode_solver(Parameters::AllParameters parameters_input) const;

    Parameters::AllParameters reinit_params(double rewienski_b) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif //PHILIP_1D_BURGERS_REWIENSKI_FD_SENSITIVITY_H
