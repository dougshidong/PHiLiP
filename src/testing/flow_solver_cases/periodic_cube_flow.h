#ifndef __PERIODIC_CUBE_FLOW_H__
#define __PERIODIC_CUBE_FLOW_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "testing/tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include "testing/flow_solver.h"

// for generate_grid
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
class PeriodicCubeFlow : public FlowSolver<dim, nstate>
{
public:
    /// Constructor.
    PeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicCubeFlow() {};

protected:
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_volume; ///< Domain volume

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.

    /// Displays the flow setup parameters
    void display_flow_solver_setup() const override;

    /// Virtual function to generate the grid
    void generate_grid(std::shared_ptr<Triangulation> grid) const override;

    /// Virtual function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const override;

    /// Integrates over the entire domain
    double integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const;
    /// Kinetic energy integrand used for integrating over the entire domain
    double integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const;
    /// Integrand for computing the L2-error of the initialization with the initial condition
    double integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const;
};

} // Tests namespace
} // PHiLiP namespace
#endif