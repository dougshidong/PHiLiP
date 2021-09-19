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


#if PHILIP_DIM==3
    template class InitialConditionFunction_FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

