#include <deal.II/base/function.h>
#include "exact_solution.h"

namespace PHiLiP {

// ========================================================
// ZERO -- Returns zero everywhere; used a placeholder when no exact solution is defined.
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_Zero<dim,nstate,real>
::ExactSolutionFunction_Zero(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_Zero<dim,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 0;
    return value;
}

// ========================================================
// 1D SINE -- Exact solution for advection_explicit_time_study
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_1DSine<dim,nstate,real>
::ExactSolutionFunction_1DSine (double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_1DSine<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    double x_adv_speed = 1.0;

    real value = 0;
    real pi = dealii::numbers::PI;
    if(point[0] >= 0.0 && point[0] <= 2.0){
        value = sin(2*pi*(point[0] - x_adv_speed * t)/2.0);
    }
    return value;
}

// ========================================================
// Inviscid Isentropic Vortex 
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::ExactSolutionFunction_IsentropicVortex(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
    // Nothing to do here yet
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Setting constants
    const double L = 5.0; // half-width of domain
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2/gam);
    const double R = 1;
    const double sigma = 1;
    const double beta = M_infty * 5 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4; //rad

    // Centre of the vortex  at t
    const double x_travel = M_infty * t * cos(alpha);
    const double x0 = 0.0 + x_travel;
    const double y_travel = M_infty * t * sin(alpha);
    const double y0 = 0.0 + y_travel;
    const double x = std::fmod(point[0] - x0-L, 2*L)+L;
    const double y = std::fmod(point[1] - y0-L, 2*L)+L;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
    const double delta_Ux = -y/R * Omega;
    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
    const double rho = pow((1 + delta_T), 1.0/(gam-1.0));
    const double Ux = M_infty * cos(alpha) + delta_Ux;
    const double Uy = M_infty * sin(alpha) + delta_Uy;
    const double Uz = 0;
    const double p = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));

    //Convert to conservative variables
    if (istate == 0)      return rho;       //density 
    else if (istate == nstate-1) return p/(gam-1.0) + 0.5 * rho * (Ux*Ux + Uy*Uy + Uz*Uz);   //total energy
    else if (istate == 1) return rho * Ux;  //x-momentum
    else if (istate == 2) return rho * Uy;  //y-momentum
    else if (istate == 3) return rho * Uz;  //z-momentum
    else return 0;

}

//=========================================================
// FLOW SOLVER -- Exact Solution Base Class + Factory
//=========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction<dim,nstate,real>
::ExactSolutionFunction ()
    : dealii::Function<dim,real>(nstate)
{
    //do nothing
}

template <int dim, int nstate, typename real>
std::shared_ptr<ExactSolutionFunction<dim, nstate, real>>
ExactSolutionFactory<dim,nstate, real>::create_ExactSolutionFunction(
        const Parameters::FlowSolverParam& flow_solver_parameters, 
        const double time_compare)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = flow_solver_parameters.flow_case_type;
    if (flow_type == FlowCaseEnum::periodic_1D_unsteady){
        if constexpr (dim==1 && nstate==dim)  return std::make_shared<ExactSolutionFunction_1DSine<dim,nstate,real> > (time_compare);
    } else if (flow_type == FlowCaseEnum::isentropic_vortex){
        if constexpr (nstate==dim+2)  return std::make_shared<ExactSolutionFunction_IsentropicVortex<dim,nstate,real> > (time_compare);
    } else {
        // Select zero function if there is no exact solution defined
        return std::make_shared<ExactSolutionFunction_Zero<dim,nstate,real>> (time_compare);
    }
    return nullptr;
}

template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM, double>;
template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,1, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,2, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,3, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,4, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,5, double>;

} // PHiLiP namespace
