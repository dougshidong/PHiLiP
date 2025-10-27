#include "ADTypes.hpp"

#include "advection_spacetime.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> AdvectionSpacetime<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;

    const real temporal_advection = 1.0; // unit by definition
    const dealii::Tensor<1,dim-1,real> spatial_velocity_field = advection_speed();
    for (int i=0; i<nstate; ++i) {
        conv_flux[i] = 0.0;
        for (int d=0; d<dim-1; ++d) { // spatial
            conv_flux[i][d] += spatial_velocity_field[d] * solution[i];
        }
        // temporal
        conv_flux[i][dim-1] += temporal_advection * solution[i]; 
    }
    return conv_flux;
    
}
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim-1,real> AdvectionSpacetime<dim,nstate,real>
::advection_speed () const
{
    dealii::Tensor<1,dim-1,real> advection_speed;
    if(dim > 1) advection_speed[0] = this->linear_advection_velocity[0];
    if(dim > 2) advection_speed[1] = this->linear_advection_velocity[1];
    if(dim > 3) advection_speed[2] = this->linear_advection_velocity[2];
    return advection_speed;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> AdvectionSpacetime<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*solution*/,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    const dealii::Tensor<1,dim-1,real> advection_speed = this->advection_speed();
    real eig_value = 0.0;
    for (int d=0; d<dim-1; ++d) {
        eig_value += advection_speed[d]*normal[d];
    }
    for (int i=0; i<nstate; i++) {
        eig[i] = eig_value;
    }
    return eig;
}

template <int dim, int nstate, typename real>
real AdvectionSpacetime<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*soln*/) const
{
    const dealii::Tensor<1,dim-1,real> advection_speed = this->advection_speed();
    real max_eig = 0;
    for (int i=0; i<dim-1; i++) {
        real abs_adv = abs(advection_speed[i]);
        max_eig = std::max(max_eig,abs_adv);
    }
    return max_eig;
}

template class AdvectionSpacetime < PHILIP_DIM, 1, double >;
template class AdvectionSpacetime < PHILIP_DIM, 1, FadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, RadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, FadFadType>;
template class AdvectionSpacetime < PHILIP_DIM, 1, RadFadType>;
} // Physics namespace
} // PHiLiP namespace
