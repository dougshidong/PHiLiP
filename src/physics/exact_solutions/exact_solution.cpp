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
