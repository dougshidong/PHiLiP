#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "real_gas.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
RealGas<dim,nstate,real>::RealGas ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
    , gam_ref(parameters_input->euler_param.gamma_gas)
    , mach_ref(parameters_input->euler_param.mach_inf)
    , mach_ref_sqr(mach_ref*mach_ref)
    , two_point_num_flux_type(parameters_input->two_point_num_flux_type)
    , Ru(8.31446261815324) /// [J/(mol·K)]
    , MW_Air(28.9651159 * pow(10,-3)) /// [kg/mol]
    , R_Air_Dim(Ru/MW_Air) /// [J/(kg·K)] 
    , R_ref(R_Air_Dim) /// [J/(kg·K)] 
    , R_Air_NonDim(R_Air_Dim/R_ref) /// []
    , temperature_ref(298.15) /// [K]
    , u_ref(mach_ref*sqrt(gam_ref*R_Air_Dim*temperature_ref)) /// [m/s]
    , u_ref_sqr(u_ref*u_ref) /// [m/s]^2
    , tol(1.0e-10) /// []
    // TO DO: nstate-dim-1 = nspecies
{
    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
                std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>());
    // (void)real_gas_cap; // to ignore unused variable errors
    // real_gas_cap.Sp_W[real_gas_cap.i_N2]

    // for(int ispecies=0; ispecies<real_gas_cap->N_species; ispecies++) 
    // {
    //     std::cout<< real_gas_cap->Sp_name[ispecies] << ",   Molecular weight: " << real_gas_cap->Sp_WSp_W[ispecies] <<std::endl;    
    // }

    for(int ispecies=0; ispecies<2; ispecies++) 
    {
        // for (int i=0; i<9; i++)
        // {
            // std::cout<< i << std::endl;
            std::cout<< real_gas_cap->Sp_name[ispecies] << ", value: " << real_gas_cap->NASACAPCoeffs[ispecies][0][0]<<std::endl;   
        // } 
    }
    
    // std::cout<<"In constructor of real gas."<<std::endl<<std::flush;
    static_assert(nstate==dim+2+2-1, "Physics::RealGas() should be created with nstate=(PHILIP_DIM+2)+(N_SPECIES-1)"); // TO DO: UPDATE THIS with nspecies
}

template <int dim, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::cout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
    std::abort();
    return conservative_soln;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::cout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
    std::abort();
    return entropy_var;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> RealGas<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    // TO DO: define this
    std::array<real,nstate> eig;
    eig.fill(0.0);
    return eig;
}

template <int dim, int nstate, typename real>
real RealGas<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // TO DO: define this
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nstate, typename real>
real RealGas<dim,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);

    const real sound = compute_sound (conservative_soln);
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    const real max_normal_eig = abs(vel_dot_n) + sound;

    return max_normal_eig;
}

template <int dim, int nstate, typename real>
real RealGas<dim,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // zero because inviscid
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
     std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipative flux (i.e. viscous terms) for RealGas
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> RealGas<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    // nothing to add here
    std::array<real,nstate> source_term;
    source_term.fill(0.0);
    return source_term;
}

// TO DO: Provide required definition for this
// template <int dim, int nstate, typename real>
// template<typename real2>
// bool RealGas<dim,nstate,real>::check_positive_quantity(real2 &qty, const std::string qty_name) const {
//     bool qty_is_positive;
//     if (qty < 0.0) {
//         // Refer to base class for non-physical results handling
//         qty = this->template handle_non_physical_result<real2>(qty_name + " is negative.");
//         qty_is_positive = false;
//     } else {
//         qty_is_positive = true;
//     }

//     return qty_is_positive;
// }


template <int dim, int nstate, typename real>
void RealGas<dim,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // TO DO: Update this you are using any kind of BC that is not periodic
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> RealGas<dim,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const
{
    std::array<real, nstate> conservative_soln;
    for (int i=0; i<nstate; i++)
    {
        conservative_soln[i] = 1.0*primitive_soln[i];
    }

    return conservative_soln;
}

/* MAIN FUNCTIONS */
/// f_M1: mixture density
template <int dim, int nstate, typename real>
template<typename real2>
inline real2 RealGas<dim,nstate,real>
:: compute_mixture_density ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 mixture_density = conservative_soln[0];
    return mixture_density;
}

/// f_M6: species densities
template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,nstate-dim-1,real2> RealGas<dim,nstate,real>
::compute_species_densities ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 mixture_density = compute_mixture_density(conservative_soln);
    dealii::Tensor<1,nstate-dim-1,real2> species_densities;
    real2 sum = 0.0;
    for (int s=0; s<(nstate-dim-1)-1; ++s) 
        { 
            species_densities[s] = conservative_soln[dim+2+s]; 
            sum += species_densities[s];
        }
    species_densities[(nstate-dim-1)-1] = mixture_density - sum;
    return species_densities;
}

/// f_M7: mass fractions
template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,nstate-dim-1,real2> RealGas<dim,nstate,real>
::compute_mass_fractions ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 mixture_density = compute_mixture_density(conservative_soln);
    const dealii::Tensor<1,nstate-dim-1,real2> species_densities = compute_species_densities(conservative_soln);
    dealii::Tensor<1,nstate-dim-1,real2> mass_fractions;
    for (int s=0; s<(nstate-dim-1)-1; ++s) 
        { 
            mass_fractions[s] = species_densities[s]/mixture_density; 
        }
    return mass_fractions;
}

/// f_M8: mixture_from_species
template <int dim, int nstate, typename real>
template <typename real2>
inline real2 RealGas<dim,nstate,real>
::compute_mixture_from_species ( const dealii::Tensor<1,nstate-dim-1,real2> &mass_fractions, const dealii::Tensor<1,nstate-dim-1,real2> &species) const
{
    real2 mixture = 0.0; 
    for (int s=0; s<(nstate-dim-1); ++s) 
        { 
            mixture += mass_fractions[s]*species[s]; 
        }   
    return mixture;
}

/// f_M9: dimensional temperature
template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
::compute_dimensional_temperature ( const real temperature ) const
{
    const real dimensional_temperature = temperature*this->temperature_ref;
    return dimensional_temperature;
}

/// f_M9.5: species gas constants
template <int dim, int nstate, typename real>
dealii::Tensor<1,nstate-dim-1,real> RealGas<dim,nstate,real>
::compute_Rs (  const real Ru ) const
{
    dealii::Tensor<1,nstate-dim-1,real> Rs;
    for (int s=0; s<(nstate-dim-1); ++s) 
    {
        Rs[s] = Ru/real_gas_cap->Sp_W[s]/this->R_ref;
    }
    return Rs;
}

/// f_M10: species specific heat at constant pressure
template <int dim, int nstate, typename real>
dealii::Tensor<1,nstate-dim-1,real> RealGas<dim,nstate,real>
::compute_species_specific_Cp ( const real temperature ) const
{
    const real dimensional_temperature = compute_dimensional_temperature(temperature);
    dealii::Tensor<1,nstate-dim-1,real> Cp;
    const dealii::Tensor<1,nstate-dim-1,real> Rs = compute_Rs(this->Ru);
    int temperature_zone;
    /// species loop
    for (int s=0; s<(nstate-dim-1); ++s) 
        { 
            /// temperature limits if-else
            if(real_gas_cap->NASACAPTemperatureLimits[s][0]<=dimensional_temperature && dimensional_temperature<=real_gas_cap->NASACAPTemperatureLimits[s][1])
            {
                temperature_zone = 0;
            }
            else if (real_gas_cap->NASACAPTemperatureLimits[s][1]<=dimensional_temperature && dimensional_temperature<=real_gas_cap->NASACAPTemperatureLimits[s][2])
            {
                temperature_zone = 1;
            }
            else if (real_gas_cap->NASACAPTemperatureLimits[s][2]<=dimensional_temperature && dimensional_temperature<=real_gas_cap->NASACAPTemperatureLimits[s][3])
            {
                temperature_zone = 2;
            }
            else
            {
                std::cout<<"Out of NASA CAP temperature limits."<<std::endl;
                std::abort();
            }
            /// main computation
            Cp[s] = 0.0;
            for (int i=0; i<7; i++)
            {
                Cp[s] += real_gas_cap->NASACAPCoeffs[s][i][temperature_zone]*pow(dimensional_temperature,i-2);
            }
            Cp[s] *= Rs[s];
        }
    return Cp;
}

/// f_M11: species specific heat at constant volume
template <int dim, int nstate, typename real>
dealii::Tensor<1,nstate-dim-1,real> RealGas<dim,nstate,real>
::compute_species_specific_Cv ( const real temperature ) const
{
    const dealii::Tensor<1,nstate-dim-1,real> Cp = compute_species_specific_Cp(temperature);
    const dealii::Tensor<1,nstate-dim-1,real> Rs = compute_Rs(this->Ru);
    dealii::Tensor<1,nstate-dim-1,real> Cv;
    for (int s=0; s<(nstate-dim-1); ++s) 
    {
        Cv[s] = Cp[s] - Rs[s];
    }
    return Cv;
}

//// up
template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,dim,real2> RealGas<dim,nstate,real>
::compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = conservative_soln[0];
    dealii::Tensor<1,dim,real2> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}

template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    return conservative_soln[0]*0.0;
}

/// IT IS FOR ALGORITHM 7
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int i=0; i<nstate; i++)
    {
        conv_flux[i] = conservative_soln[i]*0.0;
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
dealii::Vector<double> RealGas<dim,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    std::vector<std::string> names = post_get_names ();
    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {

        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
        /*const std::array<double, nstate> primitive_soln = convert_conservative_to_primitive<real>(conservative_soln);*/
        // if (primitive_soln[0] < 0) this->pcout << evaluation_points << std::endl;

        // Mixture density
          /*computed_quantities(++current_data_index) = primitive_soln[0];*/
            computed_quantities(++current_data_index) = compute_mixture_density(conservative_soln);
        // Velocities
        for (unsigned int d=0; d<dim; ++d) {
            /*computed_quantities(++current_data_index) = primitive_soln[1+d];*/
            computed_quantities(++current_data_index) = conservative_soln[1+d]/conservative_soln[0];
        }
        // Mixture momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Mixture energy
        computed_quantities(++current_data_index) = conservative_soln[nstate-1];
        // Pressure
        /*computed_quantities(++current_data_index) = primitive_soln[nstate-1];*/
        computed_quantities(++current_data_index) = 999;
        // Pressure coefficient
        /*computed_quantities(++current_data_index) = (primitive_soln[nstate-1] - pressure_inf) / dynamic_pressure_inf;*/
        computed_quantities(++current_data_index) = 999;
        // Temperature
        /*computed_quantities(++current_data_index) = compute_temperature<real>(primitive_soln);*/
        computed_quantities(++current_data_index) = 999;     //note: missing <real>
        // Entropy generation
        /*computed_quantities(++current_data_index) = compute_entropy_measure(conservative_soln) - entropy_inf;*/
        computed_quantities(++current_data_index) = 999;
        // Mach Number
        /*computed_quantities(++current_data_index) = compute_mach_number(conservative_soln);*/
        computed_quantities(++current_data_index) = 999;
        // NASA_CAP
        computed_quantities(++current_data_index) = 999;
        // speed of sound
        computed_quantities(++current_data_index) = 999;
        // temperature dim
        computed_quantities(++current_data_index) = 999;
        // species densities
        for (unsigned int s=0; s<nstate-dim-1; ++s) 
        {
            computed_quantities(++current_data_index) = compute_species_densities(conservative_soln)[s];
        }     

    }
    if (computed_quantities.size()-1 != current_data_index) {
        this->pcout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> RealGas<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nstate,real>::post_get_data_component_interpretation (); // state variables
    interpretation.push_back (DCI::component_is_scalar); // Mixture density
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
    }
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Mixture momentum
    }
    interpretation.push_back (DCI::component_is_scalar); // Mixture energy
    interpretation.push_back (DCI::component_is_scalar); // Mixture pressure
    interpretation.push_back (DCI::component_is_scalar); // Pressure coefficient
    interpretation.push_back (DCI::component_is_scalar); // Temperature
    interpretation.push_back (DCI::component_is_scalar); // Entropy generation
    interpretation.push_back (DCI::component_is_scalar); // Mach number
    interpretation.push_back (DCI::component_is_scalar); // e_comparison
    interpretation.push_back (DCI::component_is_scalar); // Sound 
    interpretation.push_back (DCI::component_is_scalar); // Temperature (Dim)
    for (unsigned int s=0; s<nstate-dim-1; ++s) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Species densities
    }

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        this->pcout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}

template <int dim, int nstate, typename real>
std::vector<std::string> RealGas<dim,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names = PhysicsBase<dim,nstate,real>::post_get_names ();
    names.push_back ("mixture_density");
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("velocity");
    }
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("mixture_momentum");
    }
    names.push_back ("mixture_energy");
    names.push_back ("pressure");
    names.push_back ("pressure_coeffcient");
    names.push_back ("temperature");

    names.push_back ("entropy_generation");
    names.push_back ("mach_number");
    names.push_back ("e_comparison");
    names.push_back ("speed_of_sound");
    names.push_back ("dimensional_temperature");
    for (unsigned int s=0; s<nstate-dim-1; ++s) 
    {
      names.push_back ("species_densities");
    }

    return names;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags RealGas<dim,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           ;
}




// Instantiate explicitly
template class RealGas < PHILIP_DIM, PHILIP_DIM+2+2-1, double     >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2+2-1, FadType    >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2+2-1, RadType    >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2+2-1, FadFadType >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2+2-1, RadFadType >;

} // Physics namespace
} // PHiLiP namespace