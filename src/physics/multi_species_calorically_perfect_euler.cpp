#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "multi_species_calorically_perfect_euler.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>::MultiSpeciesCaloricallyPerfect ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : RealGas<dim,nspecies,nstate,real>(parameters_input,manufactured_solution_function,has_nonzero_diffusion,has_nonzero_physical_source)
    , Cp(this->compute_species_specific_Cp(298.15/this->temperature_ref))
    , Cv(this->compute_species_specific_Cv(298.15/this->temperature_ref))
{
    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
        std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>(parameters_input));

    for (int s=0; s<(nspecies); ++s) 
    {
        this->gamma[s] = Cp[s]/Cv[s];
    }

    static_assert(nstate==dim+2+(nspecies-1), "Physics::MultiSpeciesCaloricallyPerfect() should be created with nstate=(PHILIP_DIM+2)+(PHILIP_SPECIES-1)");
}

// /// f_M18: convective flux
// template <int dim, int nspecies, int nstate, typename real>
// std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>
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
//         for (int s=0; s<(nspecies)-1; ++s)
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

/// f_M14: compute_temperature
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>
::compute_temperature ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = conservative_soln[0]; // TO DO: use compute_mixture_density
    const real mixture_gas_constant = this->compute_mixture_gas_constant(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real temperature = mixture_pressure/(mixture_density*mixture_gas_constant)*(this->gam_ref*this->mach_ref_sqr);

    return temperature;
}

/// f_M16: compute_mixture_pressure
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>
::compute_mixture_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = conservative_soln[0]; // TO DO: use compute_mixture_density
    const std::array<real,nspecies> gamma = compute_species_specific_heat_ratio(conservative_soln);
    const std::array<real,nspecies> mass_fractions = this->compute_mass_fractions(conservative_soln);
    const real mixture_gamma = this->compute_mixture_from_species(mass_fractions,gamma);
    const real E = this->compute_mixture_specific_total_energy(conservative_soln);
    const real k = this->compute_specific_kinetic_energy(conservative_soln);
    const real mixture_pressure = mixture_density*(mixture_gamma*this->gam_ref-1.0)*(E-k);

    return mixture_pressure;
}

/// f_S19: primitive to conservative
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const 
{
    /* definitions */
    std::array<real, nstate> conservative_soln;
    // for (int i=0; i<nstate; i++) {conservative_soln[i] = 0.0;}
    const real mixture_density = primitive_soln[0]; // TO DO: use compute_mixture_density
    std::array<real, dim> vel;
    // for (int d=0; d<dim; ++d) { vel[d] = primitive_soln[1+d]; }

    real vel2 = 0.0;
    real sum = 0.0;
    std::array<real,nspecies> species_densities;
    std::array<real,nspecies> mass_fractions;
    const std::array<real,nspecies> gamma_s = compute_species_specific_heat_ratio(primitive_soln);
    const real mixture_pressure = primitive_soln[dim+2-1];

    /* mixture density */
    conservative_soln[0] = mixture_density;

    /* mixture momentum */
    for (int d=0; d<dim; ++d) 
    {
        vel[d] = primitive_soln[1+d];
        vel2 = vel2 + vel[d]*vel[d]; ;
        conservative_soln[1+d] = mixture_density*vel[d];
    }

    /* mixture energy */
    // mass fractions
    for (int s=0; s<(nspecies)-1; ++s) 
    { 
        mass_fractions[s] = primitive_soln[dim+2+s];
        sum += mass_fractions[s];
    }
    mass_fractions[(nspecies)-1] = 1.00 - sum;     
    // species densities
    for (int s=0; s<nspecies; ++s) 
    { 
        species_densities[s] = mixture_density*mass_fractions[s];
    }
    // mixturegas gamma
    const real mixture_gamma = this->compute_mixture_from_species(mass_fractions,gamma_s);
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*vel2;  
    // mixture energy
    const real mixture_specific_total_energy = specific_kinetic_energy + mixture_pressure/(mixture_density*(mixture_gamma*this->gam_ref-1.0));
    conservative_soln[dim+2-1] = mixture_density*mixture_specific_total_energy;

    /* species densities */
    for (int s=0; s<(nspecies)-1; ++s) 
    {
        conservative_soln[dim+2+s] = species_densities[s];
    }

    return conservative_soln;
}

/// f_M20: species specific heat ratio
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpeciesCaloricallyPerfect<dim,nspecies,nstate,real>
::compute_species_specific_heat_ratio ( const std::array<real,nstate> &/*conservative_soln*/ ) const
{
    return this->gamma;
}

// Instantiate explicitly
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double     >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, FadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, RadType    >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, FadFadType >;
template class MultiSpeciesCaloricallyPerfect < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, RadFadType >;

} // Physics namespace
} // PHiLiP namespace