#include <Sacado.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>
#include <deal.II/base/function_time.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>

#include "manufactured_solution.h"

template class dealii::FunctionTime<Sacado::Fad::DFad<double>>; // Needed by Function
template class dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>;

namespace PHiLiP {

template <int dim, typename real>
ManufacturedSolutionFunction<dim,real>
::ManufacturedSolutionFunction (const unsigned int nstate)
    :
    dealii::Function<dim,real>(nstate)
    , base_values(nstate)
    , amplitudes(nstate)
    , frequencies(nstate)
{

    const double pi = atan(1)*4.0;
    //const double ee = exp(1);

    for (int s=0; s<(int)nstate; s++) {
        base_values[s] = pi*(s+1)/nstate;
        amplitudes[s] = 0.2*base_values[s]*sin((static_cast<double>(nstate)-s)/nstate);
        for (int d=0; d<dim; d++) {
            frequencies[s][d] = 1.0+sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
        }
    }
}

template <int dim, typename real>
inline real ManufacturedSolutionFunction<dim,real>
::value (const dealii::Point<dim> &point, const unsigned int istate) const
{
    real value = amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= sin( frequencies[istate][d] * point[d] );
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient (const dealii::Point<dim> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        gradient[dim_deri] = amplitudes[istate] * frequencies[istate][dim_deri] * cos( frequencies[istate][dim_deri] * point[dim_deri] );
        for (int dim_sine=0; dim_sine<dim; dim_sine++) {
            if (dim_deri != dim_sine) gradient[dim_sine] *= sin( frequencies[istate][dim_sine] * point[dim_sine] );
        }
    }
    return gradient;
}

template class ManufacturedSolutionFunction<PHILIP_DIM,double>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Fad::DFad<double>>;

}
