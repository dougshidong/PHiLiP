#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>

// for generate_grid
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Tests {
/// Selects which flow case to simulate.
template <int dim, int nstate>
class FlowSolver : public TestsBase
{
public:
    /// Constructor.
    FlowSolver(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~FlowSolver() {};

    std::shared_ptr< InitialConditionFunction<dim,double> > initial_condition_function; ///< Initial condition function
    const Parameters::AllParameters all_param; ///< All parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters
    const double courant_friedrich_lewy_number; ///< Courant-Friedrich-Lewy (CFL) number
    const unsigned int poly_degree; ///< Polynomial order
    const double final_time; ///< Final time of solution
    const std::string unsteady_data_table_filename_with_extension; ///< Filename (with extension) for the unsteady data table
        
    /// Displays the flow setup parameters
    virtual void display_flow_solver_setup(const Parameters::AllParameters *const all_param) const;

    /// Pure virtual function to generate the grid
    virtual void generate_grid(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> grid) const = 0;

    /// Pure virtual function to compute the constant time step
    virtual double get_constant_time_step(std::shared_ptr <DGBase<dim, double>> dg) const = 0;

    /// Virtual function to compute the desired unsteady data and write it to the table
    virtual void compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const;

    /// Runs the test (i.e. flow solver)
    int run_test () const;
};

template <int dim, int nstate>
class PeriodicCubeFlow : public FlowSolver<dim, nstate>
{
public:
    /// Constructor.
    PeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~PeriodicCubeFlow() {};

    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_volume; ///< Domain volume
    
    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.

    /// Displays the flow setup parameters
    void display_flow_solver_setup(const Parameters::AllParameters *const all_param) const override;

    /// Virtual function to generate the grid
    void generate_grid(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> grid) const;

    /// Virtual function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time, 
        const std::shared_ptr <DGBase<dim, double>> dg, 
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const override;

protected:
    /// Integrates over the entire domain
    double integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const;
    /// Kinetic energy integrand used for integrating over the entire domain
    double integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const;
    /// Integrand for computing the L2-error of the initialization with the initial condition
    double integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const;
};

/// Create specified flow solver as FlowSolver object 
/** Factory design pattern whose job is to create the correct physics
 */
template <int dim, int nstate>
class FlowSolverFactory
{
public:
    /// Factory to return the correct flow solver given input file.
    static std::unique_ptr< FlowSolver<dim,nstate> >
        create_FlowSolver(const Parameters::AllParameters *const parameters_input);
};

} // Tests namespace
} // PHiLiP namespace
#endif