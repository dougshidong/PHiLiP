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

#if PHILIP_DIM==1
        using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

/// Selects which flow case to simulate.
template <int dim, int nstate>
class FlowSolver : public TestsBase
{
public:
    /// Constructor.
    FlowSolver(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~FlowSolver() {};

    /// Runs the test (i.e. flow solver)
    int run_test () const;

protected:
    std::shared_ptr< InitialConditionFunction<dim,double> > initial_condition_function; ///< Initial condition function
    const Parameters::AllParameters all_param; ///< All parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters
    const double courant_friedrich_lewy_number; ///< Courant-Friedrich-Lewy (CFL) number
    const unsigned int poly_degree; ///< Polynomial order
    const double final_time; ///< Final time of solution
    const std::string unsteady_data_table_filename_with_extension; ///< Filename (with extension) for the unsteady data table
        
    /// Displays the flow setup parameters
    virtual void display_flow_solver_setup() const;

    /// Pure virtual function to generate the grid
    virtual void generate_grid(std::shared_ptr<Triangulation> grid) const = 0;

    /// Pure virtual function to compute the constant time step
    virtual double get_constant_time_step(std::shared_ptr <DGBase<dim, double>> dg) const;

    /// Virtual function to compute the desired unsteady data and write it to the table
    virtual void compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) const;

    /// Restarts the computation from a desired outputted step
    void restart_computation_from_outputted_step(std::shared_ptr <DGBase<dim, double>> dg) const;
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