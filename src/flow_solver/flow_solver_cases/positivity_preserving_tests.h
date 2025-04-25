#ifndef __POSITIVITY_TESTS_H__
#define __POSITIVITY_TESTS_H__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"
#include "dg/dg_base.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class PositivityPreservingTests : public CubeFlow_UniformGrid<dim, nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif

 public:
    /// Constructor.
    explicit PositivityPreservingTests(const Parameters::AllParameters *const parameters_input);
     
    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

 protected:
    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step_initial;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nstate>::update_maximum_local_wave_speed;

    /// Check positivity of density and total energy + verify that density is not NaN
    void check_positivity_density(DGBase<dim, double>& dg);

    /// Updates the maximum local wave speed
    double compute_integrated_entropy(DGBase<dim, double>& dg) const;

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    using FlowSolverCaseBase<dim, nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
 
 private:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Storing entropy at first step
    double initial_entropy;

    /// Store previous entropy
    double previous_numerical_entropy;

    // euler physics pointer for computing physical quantities.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif