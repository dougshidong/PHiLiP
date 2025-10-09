#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "real_gas.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
RealGas<dim,nspecies,nstate,real>::RealGas ( 
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
    , R_ref(Ru/MW_Air) /// = Ru/MW_Air [J/(kg·K)]
    , temperature_ref(298.15) /// [K]
    , u_ref(mach_ref*sqrt(gam_ref*R_ref*temperature_ref)) /// [m/s]
    , u_ref_sqr(u_ref*u_ref) /// [m/s]^2
    , tol(1.0e-14) /// []
    , density_ref(1.225) /// [kg/m^3]
    // Note: nspecies = nspecies
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,dim+2,real> > (
            parameters_input,
            parameters_input->euler_param.ref_length,
            parameters_input->euler_param.gamma_gas,
            parameters_input->euler_param.mach_inf,
            parameters_input->euler_param.angle_of_attack,
            parameters_input->euler_param.side_slip_angle,
            parameters_input->navier_stokes_param.prandtl_number,
            parameters_input->navier_stokes_param.reynolds_number_inf,
            parameters_input->navier_stokes_param.use_constant_viscosity,
            parameters_input->navier_stokes_param.nondimensionalized_constant_viscosity,
            parameters_input->navier_stokes_param.temperature_inf,
            parameters_input->navier_stokes_param.nondimensionalized_isothermal_wall_temperature,
            parameters_input->navier_stokes_param.thermal_boundary_condition_type,
            manufactured_solution_function,
            parameters_input->two_point_num_flux_type))
{
    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
                std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>(parameters_input));
    
    // Note: modify this when you change the number of species. nstate == dim+2+nspecies-1
    static_assert(nstate==dim+2+nspecies-1, "Physics::RealGas() should be created with nstate=(PHILIP_DIM+2)+(PHILIP_SPECIES-1)"); // Note: update this with nspecies in the future
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nspecies, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::cout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
    std::abort();
    return conservative_soln;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nspecies, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::cout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
    std::abort();
    return entropy_var;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim,nspecies,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // *** ADDED BY SHRUTHI - NEEDS TO BE VALIDATED/VERIFIED ***
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    std::array<real,nstate> eig;
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    for (int i=0; i<nstate; i++) {
        eig[i] = vel_dot_n;
    }

    return eig;
}

template <int dim, int nspecies, int nstate, typename real>
real RealGas<dim,nspecies,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    // *** ADDED BY SHRUTHI - NEEDS TO BE VALIDATED/VERIFIED ***
    const real sound = compute_sound(conservative_soln);
    real vel2 = compute_velocity_squared(conservative_soln);

    const real max_eig = sqrt(vel2) + sound;

    return max_eig;
}

template <int dim, int nspecies, int nstate, typename real>
real RealGas<dim,nspecies,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // *** ADDED BY SHRUTHI - NEEDS TO BE VALIDATED/VERIFIED ***
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);

    const real sound = compute_sound (conservative_soln);

    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    const real max_normal_eig = abs(vel_dot_n) + sound;

    return max_normal_eig;
}

template <int dim, int nspecies, int nstate, typename real>
real RealGas<dim,nspecies,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // zero because inviscid
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nspecies,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
     std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipative flux (i.e. viscous terms) for this physics class
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim,nspecies,nstate,real>
::source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    std::cout<<"Source Terms not implemented for RealGas."<<std::endl;
    std::abort();
    std::array<real,nstate> source_term;
    source_term.fill(0.0);
    return source_term;
}

template <int dim, int nspecies, int nstate, typename real>
void RealGas<dim,nspecies,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Note: update this if you are using any kind of BC that is not periodic for multi-species gas
    std::cout<<"Boundary Conditions not implemented for RealGas."<<std::endl;
    std::abort();
}

// Details of the following algorithms are presented in Liki's Master's thesis.
/* MAIN FUNCTIONS */
// Algorithm 1 (f_M1): Compute mixture density
template <int dim, int nspecies, int nstate, typename real>
template<typename real2>
inline real2 RealGas<dim,nspecies,nstate,real>
:: compute_mixture_density ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 mixture_density = conservative_soln[0];

    return mixture_density;
}

// Algorithm 2 (f_M2): Compute velocities
template <int dim, int nspecies, int nstate, typename real>
inline dealii::Tensor<1,dim,real> RealGas<dim,nspecies,nstate,real>
::compute_velocities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    dealii::Tensor<1,dim,real> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/mixture_density; }

    return vel;
}

// Algorithm 3 (f_M3): Compute squared velocities
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_velocity_squared ( const std::array<real,nstate> &conservative_soln ) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + vel[d]*vel[d]; 
    }  

    return vel2;
}

// Algorithm 4 (f_M4): Compute specific kinetic energy
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_specific_kinetic_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real vel2 = compute_velocity_squared(conservative_soln);
    const real k = 0.5*vel2;

    return k;
}

// Algorithm 5 (f_M5): Compute mixture specific total energy
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_specific_total_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_specific_total_energy = conservative_soln[dim+2-1]/mixture_density;

    return mixture_specific_total_energy;
}

// Algorithm 6 (f_M6): Compute species densities
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_densities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    std::array<real,nspecies> species_densities;
    real sum = 0.0;
    for (int s=0; s<nspecies-1; ++s) 
        { 
            species_densities[s] = conservative_soln[dim+2+s]; 
            sum += species_densities[s];
        }
    species_densities[nspecies-1] = mixture_density - sum;

    return species_densities;
}

// Algorithm 7 (f_M7): Compute mass fractions
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_mass_fractions ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
    std::array<real,nspecies> mass_fractions;
    for (int s=0; s<nspecies; ++s) 
        { 
            mass_fractions[s] = species_densities[s]/mixture_density; 
        }

    return mass_fractions;
}

// Algorithm 8 (f_M8): Compute mixture from species
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_from_species ( const std::array<real,nspecies> &mass_fractions, const std::array<real,nspecies> &species) const
{
    real mixture = 0.0; 
    for (int s=0; s<nspecies; ++s) 
        { 
            mixture += mass_fractions[s]*species[s]; 
        }   

    return mixture;
}

// Algorithm 9 (f_M9): Compute dimensional temperature
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_dimensional_temperature ( const real temperature ) const
{
    const real dimensional_temperature = temperature*this->temperature_ref;

    return dimensional_temperature;
}

// Algorithm 10 (f_M10): Compute species gas constants
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_Rs ( const real Ru ) const
{
    std::array<real,nspecies> Rs;
    for (int s=0; s<nspecies; ++s) 
    {
        Rs[s] = Ru/real_gas_cap->Sp_W[s]/this->R_ref;
    }

    return Rs;
}

// Algorithm 11 (f_M11): Compute species specific heat at constant pressure
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_Cp ( const real temperature ) const
{
    const real dimensional_temperature = compute_dimensional_temperature(temperature);
    std::array<real,nspecies> Cp;
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    int temperature_zone;
    // species loop
    for (int s=0; s<nspecies; ++s) 
        { 
            // There are three temperature limits in NASA CAP database
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
            // main computation
            Cp[s] = 0.0;
            for (int i=0; i<7; i++)
            {
                Cp[s] += real_gas_cap->NASACAPCoeffs[s][i][temperature_zone]*pow(dimensional_temperature,i-2);
            }
            Cp[s] *= Rs[s];
        }

    return Cp;
}

// Algorithm 12 (f_M12): Compute species specific heat at constant volume
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_Cv ( const real temperature ) const
{
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    std::array<real,nspecies> Cv;
    for (int s=0; s<nspecies; ++s) 
    {
        Cv[s] = Cp[s] - Rs[s];
    }

    return Cv;
}

// Algorithm 13 (f_M13): Compute species specific enthalpy
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_enthalpy ( const real temperature ) const
{
    const real dimensional_temperature = compute_dimensional_temperature(temperature);
    std::array<real,nspecies>h;
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    int temperature_zone;
    /// species loop
    for (int s=0; s<nspecies; ++s) 
        { 
            // There are three temperature limits in NASA CAP database
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
            // main computation
            h[s] = -real_gas_cap->NASACAPCoeffs[s][0][temperature_zone]*pow(dimensional_temperature,-2)
                   +real_gas_cap->NASACAPCoeffs[s][1][temperature_zone]*pow(dimensional_temperature,-1)*log(dimensional_temperature) 
                   +real_gas_cap->NASACAPCoeffs[s][7][temperature_zone]*pow(dimensional_temperature,-1); // The first 2 terms and the last term are added
            for (int i=0+2; i<7; i++)
            {
                h[s] += real_gas_cap->NASACAPCoeffs[s][i][temperature_zone]*pow(dimensional_temperature,i-2)/((double)(i-1)); // The other terms are added
            }
            h[s] *= (Rs[s]*this->R_ref)*dimensional_temperature; // dimensional value
            h[s] /= this->u_ref_sqr; // non-dimensional value
        }

    return h;
}

// Algorithm 14 (f_M14): Compute species specific internal energy
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_internal_energy( const real temperature ) const
{
    const std::array<real,nspecies> h = compute_species_specific_enthalpy(temperature);
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    std::array<real,nspecies> e;
    for (int s=0; s<nspecies; ++s) 
    {
        e[s] = h[s] - (this->R_ref*this->temperature_ref/this->u_ref_sqr)* Rs[s]*temperature;
    }

    return e;
}

// Algorithm 15 (f_M15): Compute temperature
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_temperature ( const std::array<real,nstate> &conservative_soln ) const
{
    /* definitions */
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    const real specific_kinetic_energy= compute_specific_kinetic_energy(conservative_soln);
    const real mixture_gas_constant = compute_mixture_gas_constant(conservative_soln);
    const real mixture_specific_total_energy = compute_mixture_specific_total_energy(conservative_soln);

    std::array<real,nspecies> species_specific_enthalpy;
    real mixture_specific_internal_energy;
    real mixture_specific_enthalpy;

    real f;
    std::array<real,nspecies> Cv;
    real mixture_Cv;
    real f_d; // f'
    real T_npo; // T_(n+1)
    real err = 999.9;
    int itr = 0;

    /* compute temperature using the Newton-Raphson method */
    real T_n = 2.0*this->temperature_ref; // the initial guess
    do
    {
        /// 1) f(T_n)
        // mixture specific internal energy: e = E - k
        mixture_specific_internal_energy = (mixture_specific_total_energy - specific_kinetic_energy)*this->u_ref_sqr; // dimensional value
        // species specific enthalpy at T_n
        species_specific_enthalpy = compute_species_specific_enthalpy(T_n/this->temperature_ref); // non-dimensional value
        // mixture specific enthalpy at T_n
        mixture_specific_enthalpy = compute_mixture_from_species(mass_fractions,species_specific_enthalpy)*this->u_ref_sqr; // dimensional value
        // Newton-Raphson function
        f = (mixture_specific_enthalpy - mixture_gas_constant*this->R_ref* T_n) - mixture_specific_internal_energy; // dimensional value

        /// 2) f'(T_n)
        // Cv at T_n
        Cv = compute_species_specific_Cv(T_n/this->temperature_ref); // non-dimensional value
        // mixture Cv
        mixture_Cv = compute_mixture_from_species(mass_fractions,Cv)*this->R_ref; // dimensional value
        // Newton-Raphson derivative function
        f_d = mixture_Cv;

        /// 3) main part
        T_npo = T_n - f/f_d; // dimensional value
        err = abs((T_npo-T_n)/this->temperature_ref);
        itr += 1;

        // update T
        T_n = T_npo;
    }
    while (err>this->tol);
    T_n /= temperature_ref; // non-dimensional value

    return T_n;
}

// Algorithm 16 (f_M16): Compute mixture gas constant
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_gas_constant ( const std::array<real,nstate> &conservative_soln ) const
{
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    const real mixture_gas_constant = compute_mixture_from_species(mass_fractions,Rs);
    return mixture_gas_constant;
}

// Algorithm 17 (f_M17): Compute mixture pressure
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_gas_constant = compute_mixture_gas_constant(conservative_soln);
    const real temperature = compute_temperature(conservative_soln);
    const real mixture_pressure = mixture_density*mixture_gas_constant*temperature/(this->gam_ref*this->mach_ref_sqr);

    return mixture_pressure;
}

// Algorithm 18 (f_M18): Compute mixture specific total enthalpy
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_specific_total_enthalpy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_specific_total_energy = compute_mixture_specific_total_energy(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_specific_total_enthalpy = mixture_specific_total_energy + mixture_pressure/mixture_density;

    return mixture_specific_total_enthalpy;
}

// Algorithm 19 (f_M19): Compute convective flux
template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nspecies,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const  
{
    /* definitions */
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real mixture_density = compute_mixture_density(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real mixture_specific_total_enthalpy = compute_mixture_specific_total_enthalpy(conservative_soln);
    const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);

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
        for (int s=0; s<nspecies-1; ++s)
        {
             conv_flux[nstate-1+s][flux_dim] = species_densities[s]*vel[flux_dim];
        }
    }

    return conv_flux;
}

/* Supporting FUNCTIONS */
// Algorithm 20 (f_S20): Convert primitive to conservative
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> RealGas<dim,nspecies,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const 
{
    /* definitions */
    std::array<real, nstate> conservative_soln;
    const real mixture_density = compute_mixture_density(primitive_soln);
    std::array<real, dim> vel;

    real vel2 = 0.0;
    real sum = 0.0;
    std::array<real,nspecies> species_densities;
    std::array<real,nspecies> mass_fractions;
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    const real mixture_pressure = primitive_soln[dim+2-1];
    std::array<real,nspecies> species_specific_internal_energy;
    std::array<real,nspecies> species_specific_total_energy;

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
    for (int s=0; s<nspecies-1; ++s) 
    { 
        mass_fractions[s] = primitive_soln[dim+2+s];
        sum += mass_fractions[s];
    }
    mass_fractions[nspecies-1] = 1.00 - sum;     
    // species densities
    for (int s=0; s<nspecies; ++s) 
    { 
        species_densities[s] = mixture_density*mass_fractions[s];
    }
    // mixturegas constant
    const real mixture_gas_constant = compute_mixture_from_species(mass_fractions,Rs);
    // temperature
    const real temperature = mixture_pressure/(mixture_density*mixture_gas_constant) * (this->u_ref_sqr/(this->R_ref*this->temperature_ref));
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*vel2;
    // species specific enthalpy
    const std::array<real,nspecies> species_specific_enthalpy = compute_species_specific_enthalpy(temperature); 
    // species energy
    for (int s=0; s<nspecies; ++s) 
    { 
      species_specific_internal_energy[s] = species_specific_enthalpy[s] - (this->R_ref*this->temperature_ref/this->u_ref_sqr)* Rs[s]*temperature;
      species_specific_total_energy[s] =  species_specific_internal_energy[s] + specific_kinetic_energy;
    }     
    // mixture energy
    const real mixture_specific_total_energy = compute_mixture_from_species(mass_fractions,species_specific_total_energy);
    conservative_soln[dim+2-1] = mixture_density*mixture_specific_total_energy;

    /* species densities */
    for (int s=0; s<nspecies-1; ++s) 
    {
        conservative_soln[dim+2+s] = species_densities[s];
    }

    return conservative_soln;
}

// Algorithm 21 (f_S21): Compute species specific heat ratio
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_heat_ratio ( const std::array<real,nstate> &conservative_soln ) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    const std::array<real,nspecies> Cv = compute_species_specific_Cv(temperature);
    std::array<real,nspecies> gamma;

    for (int s=0; s<nspecies; ++s) 
    {
        gamma[s] = Cp[s]/Cv[s];
    }

    return gamma;
}

template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_gamma ( const std::array<real,nstate> &conservative_soln ) const
{
    // *** ADDED BY SHRUTHI - NEEDS TO BE VALIDATED/VERIFIED ***
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    const std::array<real,nspecies> Cv = compute_species_specific_Cv(temperature);

    real mixture_Cp = compute_mixture_from_species(mass_fractions,Cp);
    real mixture_Cv = compute_mixture_from_species(mass_fractions,Cv);

    real gamma = mixture_Cp/mixture_Cv;
    return gamma;
}

// Algorithm 22 (f_S22): Compute species speed of sound
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_speed_of_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> gamma = compute_species_specific_heat_ratio(conservative_soln);
    const std::array<real,nspecies> Rs = compute_Rs(this->Ru);
    std::array<real,nspecies> speed_of_sound;
    for (int s=0; s<nspecies; ++s) 
        { 
            speed_of_sound[s] = sqrt(gamma[s]*Rs[s]*temperature/(this->mach_ref_sqr)); 
        }

    return speed_of_sound;
}

template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    // *** ADDED BY SHRUTHI - NEEDS TO BE VALIDATED/VERIFIED ***
    real gam = compute_gamma(conservative_soln);

    real mixture_density = conservative_soln[0];
    // check_positive_quantity(density, "density");
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);

    const real sound = sqrt(mixture_pressure*gam/mixture_density);
    return sound;
}

// Compute mixture solution vector (without species solution)
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,dim+2> RealGas<dim,nspecies,nstate,real>
::get_mixture_solution_vector ( const std::array<real,nstate> &full_soln ) const 
{
    /* definitions */
    std::array<real, dim+2> mixture_soln;
    for (int s=0; s<(dim+2); ++s) 
    { 
        mixture_soln[s] = full_soln[s];
    }
    return mixture_soln;
}

// Compute mixture gradient
template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,dim+2> RealGas<dim,nspecies,nstate,real>
::get_mixture_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,dim+2> mixture_soln_gradient;
    for (int d1=0; d1<dim; d1++) {
        mixture_soln_gradient[0][d1] = conservative_soln_gradient[0][d1];
        for (int d2=0; d2<dim; d2++) {
            mixture_soln_gradient[1+d1][d2] = conservative_soln_gradient[1+d2][d1];
        }
        mixture_soln_gradient[dim+2-1][d1] = conservative_soln_gradient[dim+2-1][d1];
    }
    return mixture_soln_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Vector<double> RealGas<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector (
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
        // get the solution
        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
        // get mixture solution for computing quantities from Navier-Stokes object
        std::array<double,dim+2> mixture_soln = get_mixture_solution_vector(conservative_soln);
        // mixture_soln[dim+2-1] = 1.0e10; // hacky fix warning -- does not affect vorticity calc
        // get the solution gradient
        std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient;
        for (unsigned int s=0; s<nstate; ++s) {
            for (unsigned int d=0; d<dim; ++d) {
                conservative_soln_gradient[s][d] = duh[s][d];
            }
        }
        // get mixture solution gradient for computing quantities from Navier-Stokes object
        const std::array<dealii::Tensor<1,dim,double>,dim+2> mixture_soln_gradient = get_mixture_solution_gradient(conservative_soln_gradient);
        // Mixture density
        computed_quantities(++current_data_index) = compute_mixture_density(conservative_soln);
        // Velocities
        const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = vel[d];
        }
        // Mixture momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Mixture energy
        computed_quantities(++current_data_index) = compute_mixture_specific_total_energy(conservative_soln);
        // Mixture pressure
        computed_quantities(++current_data_index) = compute_mixture_pressure(conservative_soln);
        // Non-dimensional temperature
        computed_quantities(++current_data_index) = compute_temperature(conservative_soln); 
        // Dimensional temperature
        computed_quantities(++current_data_index) = compute_dimensional_temperature(compute_temperature(conservative_soln));
        // Mixture specific total enthalpy
        computed_quantities(++current_data_index) = compute_mixture_specific_total_enthalpy(conservative_soln);  
        // Mass fractions
        const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = mass_fractions[s];
        }
        // Species densities
        const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = species_densities[s];
        }
        // Vorticity
        dealii::Tensor<1,3,double> vorticity = this->navier_stokes_physics->compute_vorticity(mixture_soln,mixture_soln_gradient);
        for (unsigned int d=0; d<3; ++d) {
            computed_quantities(++current_data_index) = vorticity[d];
        }

    }
    if (computed_quantities.size()-1 != current_data_index) {
        this->pcout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> RealGas<dim,nspecies,nstate,real>
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
    interpretation.push_back (DCI::component_is_scalar); // Non-dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Mixture specific total enthalpy
    for (unsigned int s=0; s<nspecies; ++s) {
         interpretation.push_back (DCI::component_is_scalar); // Mass fractions
    }
    for (unsigned int s=0; s<nspecies; ++s) {
        interpretation.push_back (DCI::component_is_scalar); // Species densities
    }
    for (unsigned int d=0; d<3; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // vorticity
    }

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        this->pcout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<std::string> RealGas<dim,nspecies,nstate,real>
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
    names.push_back ("mixture_pressure");
    names.push_back ("temperature");
    names.push_back ("dimensional_temperature");
    names.push_back ("mixture_specific_total_enthalpy");
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_mass_fraction = "mass_fraction";
      std::string string_species_mass_fraction = string_mass_fraction + "_" + real_gas_cap->Sp_name[s];
      names.push_back (string_species_mass_fraction);
    }
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_density = "species_density";
      std::string string_species_density = string_density + "_" + real_gas_cap->Sp_name[s];
      names.push_back (string_species_density);
    }
    for (unsigned int d=0; d<3; ++d) {
        names.push_back ("vorticity");
    }

    return names;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::UpdateFlags RealGas<dim,nspecies,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values 
            | dealii::update_gradients
            | dealii::update_quadrature_points;
}

// Instantiate explicitly
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, double     >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, FadType    >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, RadType    >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, FadFadType >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1, RadFadType >;
} // Physics namespace
} // PHiLiP namespace