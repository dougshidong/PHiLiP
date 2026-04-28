#ifndef __Airfoil_3D_LES__
#define __Airfoil_3D_LES__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
#include "dg/dg_base.hpp"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nspecies, int nstate>
class Airfoil_3D_LES : public FlowSolverCaseBase<dim, nspecies,nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

public:
    /// Constructor.
    Airfoil_3D_LES(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~Airfoil_3D_LES() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, nspecies, double>> dg) const override;

    /// Will compute and print lift and drag coefficients
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, nspecies, double>> dg) const override;

protected:
    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,nspecies,dim+2,double> > navier_stokes_physics;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

public:
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const std::shared_ptr <ODE::ODESolverBase<dim, nspecies, double>> ode_solver, 
            const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
            const bool do_write_unsteady_data_table_file) override;

    /// Compute time-averaged solution for turbulent cases
    void compute_time_averaged_solution(
        const std::shared_ptr <ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
        const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
        const double time_step) override;

    /// Compute time-averaged Reynolds Stresses for turbulent cases
    void compute_Reynolds_stress(
        const std::shared_ptr <ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
        const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
        const double time_step) override;

protected:

    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

private:
    /// Compute lift
    double compute_lift(std::shared_ptr<DGBase<dim, nspecies, double>> dg) const;

    /// Compute drag
    double compute_drag(std::shared_ptr<DGBase<dim, nspecies, double>> dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
