#ifndef __MULTISPECIES_TESTS_H__
#define __MULTISPECIES_TESTS_H__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nspecies, int nstate>
class MultispeciesTests : public CubeFlow_UniformGrid<dim,nspecies,nstate>
{
public:
    /// Constructor.
    explicit MultispeciesTests(const Parameters::AllParameters *const parameters_input);

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::get_adaptive_time_step;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::get_adaptive_time_step_initial;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nspecies, nstate>::update_maximum_local_wave_speed;

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    using FlowSolverCaseBase<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const std::shared_ptr<ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
        const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
        const bool do_write_unsteady_data_table_file) override;

    const unsigned int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)

    /// Modifies the DG object to reverse the velocity of the flow for the isentropic vortex case
    virtual void modify_dg_object(std::shared_ptr <DGBase<dim, nspecies, double>> dg) const;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const;
    
private:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
