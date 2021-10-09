#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

#include "initial_condition.h"
//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
// #include "parameters/all_parameters.h"

// for FlowSolver class:
#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

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
    ~FlowSolver() {}; ///< Destructor.
    
    /// Initial condition function; Assigned in constructor
    // std::shared_ptr< InitialConditionFunction_FlowSolver<dim,double> > initial_condition_function;
    
    /// Generates the grid from the parameters
    // void get_grid() const;

    /// Initializes the solution with the initial condition // TO DO
    // void initialize_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics) const;

    /// Displays the flow setup parameters
    void display_flow_solver_setup(const Parameters::AllParameters *const param) const;

    /// Runs the test (i.e. flow solver)
    int run_test () const;

protected:
    // Not used?
    // double integrate_entropy_over_domain(DGBase<dim,double> &dg) const;
};

} // Tests namespace
} // PHiLiP namespace
#endif