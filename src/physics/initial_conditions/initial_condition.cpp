#include <deal.II/base/function.h>
#include "initial_condition.h"

namespace PHiLiP {
// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition
// ========================================================
template <int dim, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,real>
::InitialConditionFunction_TaylorGreenVortex (
    const unsigned int nstate,
    const double       gamma_gas,
    const double       mach_inf)
    : InitialConditionFunction<dim,real>(nstate)
    , gamma_gas(gamma_gas)
    , mach_inf(mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
{
    // Nothing to do here yet
}

template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0], y = point[1], z = point[2];
        
        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            value = sin(x)*cos(y)*cos(z); 
        }
        if(istate==2) {  
            // y-velocity
            value = -cos(x)*sin(y)*cos(z);
        }
        if(istate==3) {
            // z-velocity
            value = 0.0;
        }
        if(istate==4) {
            // pressure
            value = 1.0/(gamma_gas*mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2.0);
        }
    }
    return value;
}

template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);

        // convert primitive to conservative solution
        if(istate==0) value = rho; // density
        if(istate==1) value = rho*u; // x-momentum
        if(istate==2) value = rho*v; // y-momentum
        if(istate==3) value = rho*w; // z-momentum
        if(istate==4) value = p/(1.4-1.0) + 0.5*rho*(u*u + v*v + w*w); // total energy
    }

    return value;
}

template <int dim, typename real>
inline real InitialConditionFunction_TaylorGreenVortex<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}

// ========================================================
// 1D BURGERS REWIENSKI -- Initial Condition
// ========================================================
template <int dim, typename real>
InitialConditionFunction_BurgersRewienski<dim,real>
::InitialConditionFunction_BurgersRewienski (const unsigned int nstate)
        : InitialConditionFunction<dim,real>(nstate)
{
    // Nothing to do here yet
}

template <int dim, typename real>
inline real InitialConditionFunction_BurgersRewienski<dim,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 1.0;
    return value;
}

// ========================================================
// 1D BURGERS VISCOUS -- Initial Condition
// ========================================================
template <int dim, typename real>
InitialConditionFunction_BurgersViscous<dim,real>
::InitialConditionFunction_BurgersViscous (const unsigned int nstate)
        : InitialConditionFunction<dim,real>(nstate)
{
    // Nothing to do here yet
}

template <int dim, typename real>
inline real InitialConditionFunction_BurgersViscous<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    real value = 0;
    if(point[0] >= 0 && point[0] <= 0.25){
        value = sin(4*dealii::numbers::PI*point[0]);
    }
    return value;

}

//=========================================================
// FLOW SOLVER -- Initial Condition Base Class + Factory
//=========================================================
template <int dim, typename real>
InitialConditionFunction<dim,real>
::InitialConditionFunction (const unsigned int nstate)
    : dealii::Function<dim,real>(nstate)//,0.0) // 0.0 denotes initial time (t=0)
    , nstate(nstate)
{ 
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
std::shared_ptr< InitialConditionFunction<dim, real> >
InitialConditionFactory<dim,nstate, real>::create_InitialConditionFunction(
    Parameters::AllParameters const *const param, 
    int                                    n_state)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex) {
        if constexpr (dim==3) return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim, real> >(
                    n_state,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf);
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1)  return std::make_shared<InitialConditionFunction_BurgersRewienski<dim,real> > (
                    n_state);
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot) {
        if constexpr (dim==1)  return std::make_shared<InitialConditionFunction_BurgersViscous<dim,real> > (
                    n_state);
    } else if (flow_type == FlowCaseEnum::naca0012) {
        if constexpr (dim==2){
            Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
                    param->euler_param.ref_length,
                    param->euler_param.gamma_gas,
                    param->euler_param.mach_inf,
                    param->euler_param.angle_of_attack,
                    param->euler_param.side_slip_angle);
            return std::make_shared<Physics::FreeStreamInitialConditions<dim,nstate>>(euler_physics_double);
        }
    } else {
        n_state = 1*n_state; //To avoid unused parameter error
        std::cout << "Invalid Flow Case Type. You probably forgot to add it to the list of flow cases in initial_condition.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

// ========================================================
// ZERO INITIAL CONDITION
// ========================================================
template<int dim, typename real>
real InitialConditionFunction_Zero<dim, real> :: value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const 
{
    return 0.0;
}

template class InitialConditionFunction <PHILIP_DIM,double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_DIM+2, double>;
template class InitialConditionFactory <PHILIP_DIM, PHILIP_DIM, double>;
template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM,double>;
template class InitialConditionFunction_Zero <PHILIP_DIM,double>;

} // PHiLiP namespace
