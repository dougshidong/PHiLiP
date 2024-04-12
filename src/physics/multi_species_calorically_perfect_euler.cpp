#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "multi_species_calorically_perfect_euler.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
MultiSpeciesCaloricallyPerfect<dim,nstate,real>::MultiSpeciesCaloricallyPerfect ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : RealGas<dim,nstate,real>(parameters_input,manufactured_solution_function,has_nonzero_diffusion,has_nonzero_physical_source)
{
    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
        std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>());
    static_assert(nstate==dim+2+3-1, "Physics::MultiSpeciesCaloricallyPerfect() should be created with nstate=(PHILIP_DIM+2)+(N_SPECIES-1)"); // TO DO: UPDATE THIS with nspecies
}

// /// f_M18: convective flux
// template <int dim, int nstate, typename real>
// std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpeciesCaloricallyPerfect<dim,nstate,real>
// ::convective_flux (const std::array<real,nstate> &conservative_soln) const  
// {
//     /* definitions */
//     std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
//     const real test = this->template compute_specific_kinetic_energy(conservative_soln);
//     const real check = real_gas_cap->Sp_W[0];

//     // flux dimension loop; E -> F -> G
//     for (int flux_dim=0; flux_dim<dim; ++flux_dim) 
//     {
//         /* A) mixture density equations */
//         conv_flux[0][flux_dim] = 0.0*conservative_soln[flux_dim];

//         /* B) mixture momentum equations */
//         for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim)
//         {
//             conv_flux[1+velocity_dim][flux_dim] = 0.0*test;
//         }
//         conv_flux[1+flux_dim][flux_dim] += 0.0*check; // Add diagonal of pressure

//         /* C) mixture energy equations */
//         conv_flux[dim+2-1][flux_dim] = 0.0;

//         /* D) species density equations */
//         for (int s=0; s<(nstate-dim-1)-1; ++s)
//         {
//              conv_flux[nstate-1+s][flux_dim] = 0.0;
//         }
//     }

//     // for (int i=0; i<nstate; i++)
//     // {
//     //     conv_flux[i] = conservative_soln[i]*0.0;
//     // }
//     return conv_flux;
// }

// Instantiate explicitly
// TO DO: Modify this when you change number of species
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, double     >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, FadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, RadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, FadFadType >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_DIM+2+3-1, RadFadType >;

} // Physics namespace
} // PHiLiP namespace