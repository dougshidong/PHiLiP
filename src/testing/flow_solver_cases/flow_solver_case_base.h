#ifndef __FLOW_SOLVER_CASE_BASE__
#define __FLOW_SOLVER_CASE_BASE__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
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
    ~FlowSolverCaseBase() {};

    /// Displays the flow setup parameters
    void display_flow_solver_setup(std::shared_ptr<InitialConditionFunction<dim,nstate,double>> initial_condition) const;

    /// Pure Virtual function to generate the grid
    virtual std::shared_ptr<Triangulation> generate_grid() const = 0;

    /// Set higher order grid
    virtual void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const;

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

    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters(std::shared_ptr<InitialConditionFunction<dim,nstate,double>> initial_condition) const = 0;

private:
    /// Returns the pde type string from the all_param class member
    std::string get_pde_string() const;

    /// Returns the flow case type string from the all_param class member
    std::string get_flow_case_string() const;

};

} // Tests namespace
} // PHiLiP namespace

#endif
