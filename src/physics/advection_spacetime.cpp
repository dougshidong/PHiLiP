#include "ADTypes.hpp"

#include "advection_spacetime.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> AdvectionSpacetime<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    std::array<dealii::Tensor<1,dim-1,real>,nstate> conv_flux_spatial;

    // get spatial part from base physics
    conv_flux_spatial = ConvectionDiffusion<dim-1,nstate,real>::convective_flux(solution);

    const real temporal_advection = 1.0; // unit by definition
    for (int i=0; i<nstate; ++i) {
        conv_flux[i] = 0.0;
        for (int d=0; d<dim-1; ++d) { // spatial
            conv_flux[i][d] = conv_flux_spatial[i][d];
        }
        // temporal
        conv_flux[i][dim] = temporal_advection * solution[i]; 
    }
    return conv_flux;
    
}

template class AdvectionSpacetime < PHILIP_DIM, 1, double >;
template class AdvectionSpacetime < PHILIP_DIM, 1, FadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, RadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, FadFadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, RadFadType>;
} // Physics namespace
} // PHiLiP namespace
