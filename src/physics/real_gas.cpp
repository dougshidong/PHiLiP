#include <cmath>
#include <vector>
#include <fstream>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "real_gas.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
RealGas<dim,nspecies,nstate,real>::RealGas ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,nspecies,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nspecies,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
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
{
    // Note: modify this when you change the number of species. nstate == dim+2+nspecies-1
    static_assert(nstate==dim+nspecies+1, "Physics::RealGas() should be created with nstate=(PHILIP_DIM+2)+(PHILIP_SPECIES-1)"); // Note: update this with nspecies in the future
    if(parameters_input->chemistry_input_file=="") {
        this->pcout << "Name of chemistry file containing NASA CAP data for species has not been passed in. Aborting..." << std::endl;
        std::abort(); 
    }
    readspecies(parameters_input->chemistry_input_file);
}

// Read chemistry file
template <int dim, int nspecies, int nstate, typename real>
void RealGas<dim, nspecies, nstate, real>
::readspecies(std::string NASADataFilename)
{
    std::string line, dum_char;

    std::ifstream chemfile (NASADataFilename);
    // std::cout << "Reading NASA Coefficients and Polynomials (CAP) Data..." << std::endl;
    std::getline(chemfile, line);
    int N_species = (int)std::stof(line);
    if(nspecies != N_species) {
        std::cout << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Number of species in chemistry file does not match PHILIP_SPECIES." << std::endl
                  << "Number of species in file = " << N_species << " and PHILIP_SPECIES = " << PHILIP_SPECIES << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
    }
    // std::cout << "Running simulation with " << N_species << " species..." << std::endl;

    std::string dummy_name;
    std::string::size_type sz1;
    //===============================================
    /*-------------------------------------------
     *           SPECIES SECTION
     *-------------------------------------------*/
    for(int i=0; i<nspecies; i++)
    {
        // Init
        sz1 = 0;
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_name[i] = line;

        std::getline(chemfile, line);
        species_weight[i] = std::stof(line); // Species molecular weight [g/mol]
        species_weight[i] /= 1000.0; // Species molecular weight [kg/mol]

        std::getline(chemfile, line);
        species_enthalpy_offset[i] = std::stof(line); // Species enthalpy from T = 0 to T= 298.15K [J/mol]
        species_enthalpy_offset[i] /= (this->species_weight[i]*this->u_ref_sqr); // nondimensionalized mass value

        std::getline(chemfile, line);
        for(int j=0; j<4; j++)
        {
            line = line.substr(sz1);
            sz1 = 0;
            NASACAPTemperatureLimits[i][j] = std::stof(line,&sz1);
        }

        // Init
        for(int k=0; k<3; k++) {
            sz1 = 0;
            std::getline(chemfile, line);
            for(int j=0; j<9; j++)
            {
                line = line.substr(sz1);
                sz1 = 0;
                NASACAPCoeffs[i][j][k] = std::stod(line,&sz1);
            }
        }
    }

    this->Rs = compute_Rs(this->Ru);
}

// Get the temperature index of the species
template <int dim, int nspecies, int nstate, typename real>
std::array<int,nspecies>  RealGas<dim, nspecies, nstate, real>
::GetNASACAP_TemperatureIndex( const real temperature) const
{
    if (temperature != temperature) {
        std::cout<<"Temperature passed in is NaN...Aborting." << std::endl;
        std::abort();
    }
    if (temperature < 0) {
        std::cout<<"Temperature passed in is negative... Temperature = " << temperature << "...Aborting." << std::endl;
        std::abort();
    }
    std::array<int,nspecies> species_tempindex;
	for(int ispecies=0; ispecies<nspecies; ispecies++)
	{
		species_tempindex[ispecies] = -2; // initialize to value with no meaning
        if(temperature < NASACAPTemperatureLimits[ispecies][0]) {
			species_tempindex[ispecies] = -1; // clip to lower bound
        }
		else if((temperature >= NASACAPTemperatureLimits[ispecies][0]) && (temperature < NASACAPTemperatureLimits[ispecies][1]))
		{
			species_tempindex[ispecies] = 0; // low temp
		}
		else if((temperature >= NASACAPTemperatureLimits[ispecies][1]) && (temperature < NASACAPTemperatureLimits[ispecies][2]))
		{
			species_tempindex[ispecies] = 1; // mid temp
		}
		else if((temperature >= NASACAPTemperatureLimits[ispecies][2]) && (temperature <= NASACAPTemperatureLimits[ispecies][3]))
		{
			species_tempindex[ispecies] = 2; // high temp
		}
        else if(temperature > NASACAPTemperatureLimits[ispecies][2]) {
			species_tempindex[ispecies] = 3; // clip to higher bound
        }
		else
		{
			std::cout<<"Invalid temperature of " << temperature << " was passed in...Aborting." << std::endl;
            std::abort();
		}
	}

    return species_tempindex;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nspecies, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    this->pcout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
    std::abort();
    return conservative_soln;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> RealGas<dim, nspecies, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    this->pcout<<"Entropy variables for RealGas hasn't been done yet."<<std::endl;
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
    real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);

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
    this->pcout<<"Source Terms not implemented for RealGas."<<std::endl;
    std::abort();
    std::array<real,nstate> source_term;
    source_term.fill(0.0);
    return source_term;
}

template <int dim, int nspecies, int nstate, typename real>
void RealGas<dim,nspecies,nstate,real>
::boundary_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary for Euler
    boundary_slip_wall(normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
}

template <int dim, int nspecies, int nstate, typename real>
void RealGas<dim,nspecies,nstate,real>
::boundary_slip_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary conditions (No penetration)
    // Given by Algorithm II of the following paper
    // Krivodonova, L., and Berger, M.,
    // “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
    // Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
    const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(soln_int);

    // Copy density and pressure and mass fractions
    std::array<real,nstate> primitive_boundary_values;
    primitive_boundary_values[0] = primitive_interior_values[0];
    primitive_boundary_values[dim+1] = primitive_interior_values[dim+1];
    for (int ispecies = 0; ispecies < nspecies-1; ++ispecies) {
        primitive_boundary_values[dim+2+ispecies] = primitive_interior_values[dim+2+ispecies];
    }

    const dealii::Tensor<1,dim,real> surface_normal = -normal_int;
    dealii::Tensor<1,dim,real> velocities_int;
    for (int d=0; d<dim; d++) { velocities_int[d] = primitive_interior_values[1+d]; }
    //const dealii::Tensor<1,dim,real> velocities_bc = velocities_int - 2.0*(velocities_int*surface_normal)*surface_normal;
    real vel_int_dot_normal = 0.0;
    for (int d=0; d<dim; d++) {
        vel_int_dot_normal = vel_int_dot_normal + velocities_int[d]*surface_normal[d];
    }
    dealii::Tensor<1,dim,real> velocities_bc;
    for (int d=0; d<dim; d++) {
        velocities_bc[d] = velocities_int[d] - 2.0*(vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] = velocities_int[d] - (vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] += velocities_int[d] * surface_normal.norm_square();
    }
    for (int d=0; d<dim; ++d) {
        primitive_boundary_values[1+d] = velocities_bc[d];
    }

    const std::array<real,nstate> modified_conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate=0; istate<nstate; ++istate) {
        soln_bc[istate] = modified_conservative_boundary_values[istate];
    }

    for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = -soln_grad_int[istate];
    }
}

template <int dim, int nspecies, int nstate, typename real>
void RealGas<dim,nspecies,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1001) {
        // Wall boundary condition (slip for Euler, no-slip for Navier-Stokes; done through polymorphism)
        boundary_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } else if (boundary_type == 1006) {
        // Slip wall boundary condition
        boundary_slip_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } else {
        this->pcout<<"Boundary condition #" << boundary_type << " not implemented for RealGas."<<std::endl;
        std::abort();
    }
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
::compute_velocity_squared_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + vel[d]*vel[d]; 
    }  

    return vel2;
}

template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real> &velocities ) const
{
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + velocities[d]*velocities[d]; 
    }  

    return vel2;
}

template <int dim, int nspecies, int nstate, typename real>
inline dealii::Tensor<1,dim,real> RealGas<dim,nspecies,nstate,real>
::extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const
{
    dealii::Tensor<1,dim,real> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}

// Algorithm 4 (f_M4): Compute specific kinetic energy
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_specific_kinetic_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);
    const real k = 0.5*vel2;

    return k;
}

// Algorithm 5 (f_M5): Compute mixture specific total energy
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_specific_total_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_specific_total_energy = conservative_soln[dim+1]/mixture_density;

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
        Rs[s] = Ru/this->species_weight[s]/this->R_ref;
    }

    return Rs;
}

// Algorithm 11 (f_M11): Compute species specific heat at constant pressure
// This function has been modified by Shruthi
// Modification: separates the temperature index into its own separate function since two different functions use it
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_Cp ( const real temperature ) const
{
    real dimensional_temperature = compute_dimensional_temperature(temperature);
    std::array<real,nspecies> Cp;
    // const std::array<real,nspecies> Rs = compute_Rs(this->Ru);

    if (dimensional_temperature < 0) {
        std::cout<<"Cp Calculation Error: Temperature passed in is negative... Temperature = " << dimensional_temperature << "...Aborting." << std::endl;
        std::abort();
    }
    std::array<int,nspecies> species_tempindex = GetNASACAP_TemperatureIndex(dimensional_temperature);
    // species loop
    for (int s=0; s<nspecies; ++s) 
    { 
        // main computation
        Cp[s] = 0.0;
        if(species_tempindex[s] == -1) { // clip to lower temperature bound's Cp (Refer to NASA FUN3D manual v14.2 sec.B.8)
            species_tempindex[s] = 0;
            dimensional_temperature = NASACAPTemperatureLimits[s][0];
        }
        if(species_tempindex[s] == 3) { // clip to higher temperature bound's Cp (Refer to NASA FUN3D manual v14.2 sec.B.8)
            species_tempindex[s] = 2;
            dimensional_temperature = NASACAPTemperatureLimits[s][2];
        }
        for (int i=0; i<7; i++)
        {
            Cp[s] += this->NASACAPCoeffs[s][i][species_tempindex[s]]*pow(dimensional_temperature,i-2);
        }
        Cp[s] *= this->Rs[s];
    }

    return Cp; // nondimensional mass value
}

// Algorithm 12 (f_M12): Compute species specific heat at constant volume
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_Cv ( const real temperature ) const
{
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    std::array<real,nspecies> Cv;
    // const std::array<real,nspecies> Rs = compute_Rs(this->Ru);

    for (int s=0; s<nspecies; ++s) 
    {
        Cv[s] = Cp[s] - this->Rs[s];
    }

    return Cv; // nondimensional mass value
}

// Algorithm 13 (f_M13): Compute species specific enthalpy
// This function has been modified by Shruthi
// Modification: separates the temperature index into its own separate function since two different functions use it
// Modification #2: includes a clipping process to ensure we can still calculate for temps outside range
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> RealGas<dim,nspecies,nstate,real>
::compute_species_specific_enthalpy ( const real temperature ) const
{
    real dimensional_temperature = compute_dimensional_temperature(temperature);
    std::array<real,nspecies>h;
    
    if (dimensional_temperature < 0) {
        std::cout<<"Enthalpy Calculation Error: Temperature passed in is negative... Temperature = " << dimensional_temperature << "...Aborting." << std::endl;
        std::abort();
    }
    std::array<int,nspecies> species_tempindex = GetNASACAP_TemperatureIndex(dimensional_temperature);
    /// species loop
    for (int s=0; s<nspecies; ++s) 
    { 
        // main computation
        real Cp = 0.0;
        real out_of_bounds_temp = -1.0;
        if(species_tempindex[s] == -1) { // Calculate enthalpy using calorically perfect gas (CPG) model (Refer to NASA FUN3D manual v14.2 sec.B.8)
            species_tempindex[s] = 0;
            std::array<real,nspecies> Cp_species = compute_species_specific_Cp(NASACAPTemperatureLimits[s][0]);
            Cp = Cp_species[s]; // obtain Cp so the enthalpy can be calculated with CPG model
            Cp /= this->Rs[s]; // nondimensional molar value of Cp;
            out_of_bounds_temp = dimensional_temperature; // save the temperature value to calculate enthalpy using CPG model
            dimensional_temperature = NASACAPTemperatureLimits[s][0];
        }
        if(species_tempindex[s] == 3) { // Calculate enthalpy using calorically perfect gas (CPG) model (Refer to NASA FUN3D manual v14.2 sec.B.8)
            species_tempindex[s] = 2;
            std::array<real,nspecies> Cp_species = compute_species_specific_Cp(NASACAPTemperatureLimits[s][2]);
            Cp = Cp_species[s]; // obtain Cp so the enthalpy can be calculated with CPG model
            Cp /= this->Rs[s]; // nondimensional molar value of Cp;
            out_of_bounds_temp = dimensional_temperature; // save the temperature value to calculate enthalpy using CPG model
            dimensional_temperature = NASACAPTemperatureLimits[s][2];
        }
        h[s] = -this->NASACAPCoeffs[s][0][species_tempindex[s]]*pow(dimensional_temperature,-2)
                +this->NASACAPCoeffs[s][1][species_tempindex[s]]*pow(dimensional_temperature,-1)*log(dimensional_temperature) 
                +this->NASACAPCoeffs[s][7][species_tempindex[s]]*pow(dimensional_temperature,-1); // The first 2 terms and the last term are added
        for (int i=2; i<7; i++)
        {
            h[s] += this->NASACAPCoeffs[s][i][species_tempindex[s]]*pow(dimensional_temperature,i-2)/((double)(i-1)); // The other terms are added
        }

        if(out_of_bounds_temp != -1.0) {
            h[s] = h[s]*(dimensional_temperature/out_of_bounds_temp) + ((out_of_bounds_temp - dimensional_temperature)/out_of_bounds_temp) * Cp;
        }

        if(out_of_bounds_temp != -1.0)
            h[s] *= ((this->Ru*out_of_bounds_temp)/(this->species_weight[s]*this->u_ref_sqr)); //nondimensional mass value
        else
            h[s] *= ((this->Ru*dimensional_temperature)/(this->species_weight[s]*this->u_ref_sqr)); //nondimensional mass value
        
        h[s] += species_enthalpy_offset[s]; // add the species_enthalpy_offset to account for enthalpy of formation for T=0 -> T=298.15K

        // set dimensional temp back to the out of bounds temp for the next species in the loop
        if (out_of_bounds_temp != -1.0)
            dimensional_temperature = out_of_bounds_temp;
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
    // std::cout << std::endl;
    do
    {
        /// 1) f(T_n)
        // mixture specific internal energy: e = E - k
        mixture_specific_internal_energy = (mixture_specific_total_energy - specific_kinetic_energy)*this->u_ref_sqr; // dimensional value
        // species specific enthalpy at T_n
        species_specific_enthalpy = compute_species_specific_enthalpy(T_n/this->temperature_ref); // nondimensional mass value
        // mixture specific enthalpy at T_n
        mixture_specific_enthalpy = compute_mixture_from_species(mass_fractions,species_specific_enthalpy)*this->u_ref_sqr; // dimensional value
        // Newton-Raphson function
        f = (mixture_specific_enthalpy - mixture_gas_constant*this->R_ref* T_n) - mixture_specific_internal_energy; // dimensional value

        /// 2) f'(T_n)
        // Cv at T_n
        Cv = compute_species_specific_Cv(T_n/this->temperature_ref); // nondimensional mass value

        // mixture Cv
        mixture_Cv = compute_mixture_from_species(mass_fractions,Cv)*this->R_ref; // dimensional value

        // Newton-Raphson derivative function
        f_d = mixture_Cv;

        /// 3) main part
        T_npo = T_n - f/f_d; // dimensional value
        err = abs((T_npo-T_n)/this->temperature_ref);
        itr += 1;

        // update T
        if(itr > 9.99999e6) {
                // output temperature values for the last 10 iterations
                // included this output so user can determine if the tolerance is the issue
                std::cout << "Nearing the max iterations...iteration #" << itr << " old temperature:  " << T_n 
                            << " new temperature:  " << T_npo << std::endl;
                std::cout << " Mixture Cv:  " << mixture_Cv << std::endl << std::endl;
        }
        T_n = T_npo;
    }
    while (err>this->tol && itr < 1e7);
    if(itr == 1e7) {
        std::cout << "Maximum iterations for temperature reached without converging...Aborting..." << std::endl;
        std::abort();
    }
    // std::cout << std::endl << "next loop: " << std::endl;
    T_n /= temperature_ref; // non-dimensional value
    if(T_n < 0) {
        std::cout << "Computed temperature is a negative value...Aborting..." << std::endl;
        std::abort();
    }
    if(T_n != T_n) {
        std::cout << "Computed temperature is NaN...Aborting..." << std::endl;
        std::abort();
    }
    return T_n;
}

// Algorithm 16 (f_M16): Compute mixture gas constant
template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_mixture_gas_constant ( const std::array<real,nstate> &conservative_soln ) const
{
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    const real mixture_gas_constant = compute_mixture_from_species(mass_fractions,this->Rs);
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

template <int dim, int nspecies, int nstate, typename real>
inline real RealGas<dim,nspecies,nstate,real>
::compute_pressure_from_density_temperature ( const real density, const real temperature, const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_gas_constant = compute_mixture_gas_constant(conservative_soln);
    const real mixture_pressure = density*mixture_gas_constant*temperature/(this->gam_ref*this->mach_ref_sqr);
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
        conv_flux[dim+1][flux_dim] = mixture_density*vel[flux_dim]*mixture_specific_total_enthalpy;

        /* D) species density equations */
        for (int s=0; s<nspecies-1; ++s)
        {
             conv_flux[dim+2+s][flux_dim] = species_densities[s]*vel[flux_dim];
        }
    }

    return conv_flux;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,nstate,real> RealGas<dim,nspecies,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // Real Gas version of function in Euler
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real gam = compute_gamma(conservative_soln);
    const real gamm1 = gam - 1.0;
    const real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);
    const real phi = 0.5*gamm1 * vel2;

    const real density = conservative_soln[0];
    const real tot_energy = conservative_soln[dim+1];
    const real E = tot_energy / density;
    const real a1 = gam*E-phi;
    const real a2 = gamm1;
    const real a3 = gam-2.0;

    dealii::Tensor<2,nstate,real> jacobian;
    for (int d=0; d<dim; ++d) {
        jacobian[0][1+d] = normal[d];
    }
    for (int row_dim=0; row_dim<dim; ++row_dim) {
        jacobian[1+row_dim][0] = normal[row_dim]*phi - vel[row_dim] * vel_normal;
        for (int col_dim=0; col_dim<dim; ++col_dim){
            if (row_dim == col_dim) {
                jacobian[1+row_dim][1+col_dim] = vel_normal - a3*normal[row_dim]*vel[row_dim];
            } else {
                jacobian[1+row_dim][1+col_dim] = normal[col_dim]*vel[row_dim] - a2*normal[row_dim]*vel[col_dim];
            }
        }
        jacobian[1+row_dim][dim+1] = normal[row_dim]*a2;
    }
    jacobian[dim+1][0] = vel_normal*(phi-a1);
    for (int d=0; d<dim; ++d){
        jacobian[dim+1][1+d] = normal[d]*a1 - a2*vel[d]*vel_normal;
    }
    jacobian[dim+1][dim+1] = gam*vel_normal;

    return jacobian;
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
    const real mixture_pressure = primitive_soln[dim+1];

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
    const real mixture_gas_constant = compute_mixture_from_species(mass_fractions,this->Rs);
    // temperature
    const real temperature = mixture_pressure/(mixture_density*mixture_gas_constant) * (this->u_ref_sqr/(this->R_ref*this->temperature_ref));
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*vel2;
    // species specific enthalpy
    const std::array<real,nspecies> species_specific_enthalpy = compute_species_specific_enthalpy(temperature); 
    // mixture enthalpy
    const real mixture_specific_enthalpy = compute_mixture_from_species(mass_fractions,species_specific_enthalpy);
    // mixture specific internal energy
    const real mixture_specific_internal_energy = mixture_specific_enthalpy - mixture_pressure/mixture_density;
    // mixture specific total energy
    const real mixture_specific_total_energy = mixture_specific_internal_energy + specific_kinetic_energy;

    // mixture energy
    conservative_soln[dim+1] = mixture_density*mixture_specific_total_energy;

    /* species densities */
    for (int s=0; s<nspecies-1; ++s) 
    {
        conservative_soln[dim+2+s] = species_densities[s];
    }

    return conservative_soln;
}

// Algorithm 20b : Convert conservative to primitive
// This function has been added by Shruthi
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> RealGas<dim,nspecies,nstate,real>
::convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const 
{
    /* definitions */
    std::array<real, nstate> primitive_soln;
    primitive_soln[0] = conservative_soln[0];

    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    for (int idim = 0; idim < dim; ++idim) {
        primitive_soln[idim+1] = vel[idim];
    }

    primitive_soln[dim+1] = compute_mixture_pressure(conservative_soln);

    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    for(int ispecies = 0; ispecies < nspecies-1; ++ispecies) {
        primitive_soln[dim+2+ispecies] = mass_fractions[ispecies];
    }

    return primitive_soln;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nspecies,nstate,real>
::convert_primitive_gradient_to_conservative_gradient (
    const std::array<real,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    this->pcout << "WARNING: convert_primitive_gradient_to_conservative_gradient() is not defined for current physics." << std::endl;
    this->pcout << "Aborting..." << std::endl;
    std::abort();
    return primitive_soln_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nspecies,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    this->pcout << "WARNING: convert_conservative_gradient_to_primitive_gradient() is not defined for current physics." << std::endl;
    this->pcout << "Aborting..." << std::endl;
    std::abort();
    return conservative_soln_gradient;
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
    // Uses the definition given in Gouasmi thesis
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
    // This is the appropriate method for deriving mixture
    // speed of sound for thermally perfect gas as per
    // Hypersonic and High Temperature Gas Dynamics, 2nd Ed.
    // John D. Anderson
    // Chapter 14.7 Eqn 14.53
    const real R_mix = compute_mixture_gas_constant(conservative_soln);
    const real temperature = compute_temperature(conservative_soln);
    const real gamma = compute_gamma(conservative_soln);

    const real sound = sqrt(gamma*R_mix*temperature/(this->mach_ref_sqr)); 

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
        mixture_soln_gradient[dim+1][d1] = conservative_soln_gradient[dim+1][d1];
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
    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {
        // get the solution
        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
        
        // get the solution gradient
        std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient;
        for (unsigned int s=0; s<nstate; ++s) {
            for (unsigned int d=0; d<dim; ++d) {
                conservative_soln_gradient[s][d] = duh[s][d];
            }
        }

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
    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nspecies,nstate,real>::post_get_data_component_interpretation (); // state variables
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
    std::vector<std::string> names = PhysicsBase<dim,nspecies,nstate,real>::post_get_names ();
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
      std::string string_species_mass_fraction = string_mass_fraction + "_" + this->species_name[s];
      names.push_back (string_species_mass_fraction);
    }
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_density = "species_density";
      std::string string_species_density = string_density + "_" + this->species_name[s];
      names.push_back (string_species_density);
    }

    return names;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::UpdateFlags RealGas<dim,nspecies,nstate,real>
::post_get_needed_update_flags () const
{
    return dealii::update_values 
            | dealii::update_gradients
            | dealii::update_quadrature_points;
}

// Instantiate explicitly
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, double     >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, FadType    >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, RadType    >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, FadFadType >;
template class RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, RadFadType >;
} // Physics namespace
} // PHiLiP namespace