#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition.h"
#include "tests.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

// for generate_grid
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

// for the grid:
#include "grid_refinement_study.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver_cases/periodic_cube_flow.h"
#include "flow_solver_cases/1D_burgers_rewienski_snapshot.h"
#include "flow_solver_cases/1d_burgers_viscous_snapshot.h"
#include <deal.II/base/table_handler.h>


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
    FlowSolver(const Parameters::AllParameters *const parameters_input, std::shared_ptr<FlowSolverCaseBase<dim, nstate>>);
    
    /// Destructor
    ~FlowSolver() {};

    /// Pointer to Flow Solver Case
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case;

    /// Runs the test (i.e. flow solver)
    int run_test () const;

protected:
    std::shared_ptr< InitialConditionFunction<dim,double> > initial_condition_function; ///< Initial condition function
    const Parameters::AllParameters all_param; ///< All parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters
    const unsigned int poly_degree; ///< Polynomial order
    const double final_time; ///< Final time of solution

public:
    /// Pointer to dg so it can be accessed externally.
    std::shared_ptr<DGBase<dim, double>> dg;

    /// Pointer to ode solver so it can be accessed externally.
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

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