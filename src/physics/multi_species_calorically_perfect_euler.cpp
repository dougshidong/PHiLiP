#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "multi_species_calorically_perfect_euler.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
MultiSpeciesCaloricallyPerfect<dim,nstate,real>::MultiSpeciesCaloricallyPerfect ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
{
    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
                std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>());
    static_assert(nstate==dim+2+3-1, "Physics::MultiSpeciesCaloricallyPerfect() should be created with nstate=(PHILIP_DIM+2)+(N_SPECIES-1)"); // TO DO: UPDATE THIS with nspecies
}

/// f_M18: convective flux
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpeciesCaloricallyPerfect<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const  
{
    /* definitions */
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real mixture_density = compute_mixture_density(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real mixture_specific_total_enthalpy = compute_mixture_specific_total_enthalpy(conservative_soln);
    const std::array<real,nstate-dim-1> species_densities = compute_species_densities(conservative_soln);

    // flux dimension loop; E -> F -> G
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) 
    {
        /* A) mixture density equations */
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];

        /* B) mixture momentum equations */
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim)
        {
            conv_flux[1+velocity_dim][flux_dim] = mixture_density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += mixture_pressure; // Add diagonal of pressure

        /* C) mixture energy equations */
        conv_flux[dim+2-1][flux_dim] = mixture_density*vel[flux_dim]*mixture_specific_total_enthalpy;

        /* D) species density equations */
        for (int s=0; s<(nstate-dim-1)-1; ++s)
        {
             conv_flux[nstate-1+s][flux_dim] = species_densities[s]*vel[flux_dim];
        }
    }

    // for (int i=0; i<nstate; i++)
    // {
    //     conv_flux[i] = conservative_soln[i]*0.0;
    // }
    return conv_flux;
}

// Instantiate explicitly
// TO DO: Modify this when you change number of species
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, double     >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, FadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, RadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, FadFadType >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, RadFadType >;

} // Physics namespace
} // PHiLiP namespace