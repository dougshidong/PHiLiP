#include "flow_solver.h"

namespace PHiLiP {
namespace Tests {

// done
template <int dim, typename real>
InitialConditionFunction_FlowSolver<dim,real>
::InitialConditionFunction_FlowSolver (const unsigned int nstate)
    :
    dealii::Function<dim,real>(nstate,0.0) // 0.0 denotes initial time (t=0)
    , nstate(nstate)
{ 
    // Nothing to do here yet
}
// done
template <int dim, typename real>
std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> > 
InitialConditionFactory_FlowSolver<dim,real>::create_InitialConditionFunction_FlowSolver(
    Parameters::AllParameters const *const param, 
    int                                    nstate)
{
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;

    return create_InitialConditionFunction_FlowSolver(flow_type, nstate);
}
// done
template <int dim, typename real>
std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> >
InitialConditionFactory_FlowSolver<dim,real>::create_InitialConditionFunction_FlowSolver(
    Parameters::FlowSolverParam::FlowCaseType flow_type,
    int                                       nstate)
{
    if(flow_type == FlowCaseEnum::inviscid_taylor_green_vortex){
        if constexpr((dim==3) /*&& (nstate==dim+2)*/) {
            return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,real>>(nstate);
        }
    }else if(flow_type == FlowCaseEnum::viscous_taylor_green_vortex){
        if constexpr((dim==3) /*&& (nstate==dim+2)*/) {
            return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,real>>(nstate);
        }
    }else{
        std::cout << "Invalid Flow Case Type." << std::endl;
    }

    return nullptr;
}

template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
void FlowSolver<dim,nstate>::initialize_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics) const
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg.dof_handler, *physics.manufactured_solution_function, solution_no_ghost);
    dg.solution = solution_no_ghost;
}

// TO DO: Figure out the contents of the flow solver -- Maybe just discuss with Doug tomorrow

// Create DG from which we'll modify the HighOrderGrid
std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = 
            PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
dg->allocate_system ();

std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = 
                PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters);
initialize_perturbed_solution(*dg, *physics_double);

#if PHILIP_DIM==3
    // InitialConditionFunction
    template class InitialConditionFunction_FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

