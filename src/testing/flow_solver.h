#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

// for generate_grid
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Tests {
/// Selects which flow case to simulate.
template <int dim, int nstate>
class FlowSolver: public TestsBase
{
public:
    /// Constructor.
    FlowSolver(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~FlowSolver() {};

    std::shared_ptr< InitialConditionFunction<dim,double> > initial_condition_function; ///< Initial condition function
    double domain_left; ///< Domain left-boundary value for generating the grid
    double domain_right; ///< Domain right-boundary value for generating the grid
    double domain_volume; ///< Domain volume
    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    bool is_triply_periodic_cube = false; ///< Identifies if grid is a triply periodic cube; initialized as false.
    
    /// Displays the flow setup parameters
    void display_flow_solver_setup(const Parameters::AllParameters *const param) const;

    /// Generates the grid for the flow case from parameters
    void generate_grid(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> &grid, 
                       const int number_of_cells_per_direction) const;

    /// Computes the kinetic energy for the TGV problem -- TO DO: Move to a seperate class?
    double integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const;
    double integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const;
    double integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const;

    /// Runs the test (i.e. flow solver)
    int run_test () const;
};

} // Tests namespace
} // PHiLiP namespace
#endif