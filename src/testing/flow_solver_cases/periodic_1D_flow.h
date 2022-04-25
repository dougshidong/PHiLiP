#ifndef __PERIODIC_1D_FLOW_H__
#define __PERIODIC_1D_FLOW_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class Periodic1DFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    Periodic1DFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~Periodic1DFlow() {};

protected:
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_volume; ///< Domain volume
    const std::string unsteady_data_table_filename_with_extension; ///< Filename (with extension) for the unsteady data table

    /// Displays the flow setup parameters
    void display_flow_solver_setup() const override;

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;
    
    /// 
    //int number_of_times_refined_by_half;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const override;
};

} // Tests namespace
} // PHiLiP namespace
#endif
