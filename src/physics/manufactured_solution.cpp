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
    const double ee = exp(1);

    for (int s=0; s<nstate; s++) {
        base_values[s] = pi*(s+1)/nstate;
        for (int d=0; d<dim; d++) {
            amplitudes[s][d] = 0.5*base_values[s]*(dim-d)/dim*(nstate-s)/nstate;
            frequencies[s][d] = 1.0+sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
        }
    }
}

template <int dim, typename real>
inline real ManufacturedSolutionFunction<dim,real>
::value (const dealii::Point<dim> &point, const unsigned int istate) const
{
    real value = base_values[istate];
    for (int d=0; d<dim; d++) {
        value += amplitudes[istate][d] * sin( frequencies[istate][d] * point[d] );
    }
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,real>
::gradient (const dealii::Point<dim> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    for (int d=0; d<dim; d++) {
        gradient[d] = amplitudes[istate][d] * frequencies[istate][d] * cos( frequencies[istate][d] * point[d] );
    }
    return gradient;
}

template class ManufacturedSolutionFunction<PHILIP_DIM,double>;
template class ManufacturedSolutionFunction<PHILIP_DIM,Sacado::Fad::DFad<double>>;

}
