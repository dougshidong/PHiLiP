#ifndef __FLOW_SOLVER_ZERO__
#define __FLOW_SOLVER_ZERO__

// for FlowSolver class:
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include "dg/dg_base.hpp"
#include "flow_solver_case_base.h"
#include "parameters/all_parameters.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/physics.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nspecies, int nstate>
class FlowSolverCaseZero: public FlowSolverCaseBase<dim, nspecies, nstate>
{
public:
    /// Constructor.
    explicit FlowSolverCaseZero(const Parameters::AllParameters *const parameters_input);

    /// Virtual function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;
protected:
    /// Pointer to Physics object for computing things on the fly
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;
    
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
