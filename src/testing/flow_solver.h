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
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double final_time; ///< Final time of solution
    const double domain_volume; ///< Domain volume
    const std::string unsteady_data_table_filename; ///< Filename for the unsteady data table
    
    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    
    /// Displays the flow setup parameters
    void display_flow_solver_setup(const Parameters::AllParameters *const all_param) const;

    /// Computes the kinetic energy for the TGV problem -- TO DO: Move to a seperate class?
    double integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const;
    double integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const;
    double integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const;

    /// Computes the desired unsteady data and writes it to the table -- make this a virtual function
    void compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time, 
        const std::shared_ptr <DGBase<dim, double>> dg, 
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const;

    /// Runs the test (i.e. flow solver)
    int run_test () const;
};

} // Tests namespace
} // PHiLiP namespace
#endif