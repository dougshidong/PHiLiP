#ifndef __1D_BURGERS_REWIENSKI_SNAPSHOT__
#define __1D_BURGERS_REWIENSKI_SNAPSHOT__

// for FlowSolver class:
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include "dg/dg_base.hpp"
#include "flow_solver_case_base.h"
#include "parameters/all_parameters.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/physics.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class BurgersRewienskiSnapshot: public FlowSolverCaseBase<dim, nstate>
{
public:
    /// Constructor.
    explicit BurgersRewienskiSnapshot(const Parameters::AllParameters *const parameters_input);

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function for postprocessing when solving for steady state
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const override;

protected:
    const int number_of_refinements; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
