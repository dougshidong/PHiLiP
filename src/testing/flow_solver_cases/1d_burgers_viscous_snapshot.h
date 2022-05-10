#ifndef __1D_BURGERS_VISCOUS_SNAPSHOT__
#define __1D_BURGERS_VISCOUS_SNAPSHOT__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include "flow_solver_case_base.h"
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class BurgersViscousSnapshot: public FlowSolverCaseBase<dim, nstate>
{
public:
    /// Constructor.
    BurgersViscousSnapshot(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~BurgersViscousSnapshot() {};

    /// Virtual function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Virtual function to write unsteady snapshot data to table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

protected:
    const int number_of_refinements; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;
};

} // Tests namespace
} // PHiLiP namespace

#endif
