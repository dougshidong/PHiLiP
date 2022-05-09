#ifndef __PERIODIC_CUBE_FLOW_H__
#define __PERIODIC_CUBE_FLOW_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
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
class PeriodicCubeFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    PeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicCubeFlow() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)

    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters(std::shared_ptr<InitialConditionFunction<dim,nstate,double>> initial_condition) const override;

    /// Display grid parameters
    void display_grid_parameters() const;
};

template <int dim, int nstate>
class PeriodicTurbulence : public PeriodicCubeFlow<dim,nstate>
{
public:
    /// Constructor.
    PeriodicTurbulence(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicTurbulence() {};

    /// Computes the kinetic energy
    double compute_kinetic_energy(DGBase<dim, double> &dg) const;

protected:
    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters(std::shared_ptr<InitialConditionFunction<dim,nstate,double>> initial_condition) const override;

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.

    /// Function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const override;

    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        kinetic_energy,
        l2_error_initial_condition
    };

    /// Kinetic energy integrand used for integrating over the entire domain
    double integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const;
    /// Integrand for computing the L2-error of the initialization with the initial condition
    double integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const;

    /// Integrates over the entire domain
    double integrate_over_domain(DGBase<dim, double> &dg,const IntegratedQuantitiesEnum integrated_quantity) const;
};

} // Tests namespace
} // PHiLiP namespace
#endif
