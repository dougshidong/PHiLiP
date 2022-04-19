#ifndef __FLOW_SOLVER_CASE_BASE__
#define __FLOW_SOLVER_CASE_BASE__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
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
class FlowSolverCaseBase
{
public:
    ///Constructor
    FlowSolverCaseBase(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    virtual ~FlowSolverCaseBase() {};

    /// Displays the flow setup parameters
    virtual void display_flow_solver_setup() const;

    /// Pure Virtual function to generate the grid
    virtual std::shared_ptr<Triangulation> generate_grid() const = 0;

    /// Virtual function to write unsteady snapshot data to table
    virtual void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const;

    /// Virtual function to compute the constant time step
    virtual double get_constant_time_step(std::shared_ptr <DGBase<dim, double>> dg) const;

    /// Virtual function for postprocessing when solving for steady state
    virtual void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const;

protected:
    const Parameters::AllParameters all_param; ///< All parameters
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    const int n_mpi; ///< Number of MPI processes.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

public:
    // Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;
};

} // Tests namespace
} // PHiLiP namespace

#endif
