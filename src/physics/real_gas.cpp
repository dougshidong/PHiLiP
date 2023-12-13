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
{

    this->real_gas_cap = std::dynamic_pointer_cast<PHiLiP::RealGasConstants::AllRealGasConstants>(
                std::make_shared<PHiLiP::RealGasConstants::AllRealGasConstants>());
    // (void)real_gas_cap; // to ignore unused variable errors
    // real_gas_cap.Sp_W[real_gas_cap.i_N2]

    for(int ispecies=0; ispecies<real_gas_cap->N_species; ispecies++) {
        std::cout<< real_gas_cap->Sp_name[ispecies] << ",   Molecular weight: " << real_gas_cap->Sp_W[ispecies] <<std::endl;    
    }
    
    // std::cout<<"In constructor of real gas."<<std::endl<<std::flush;
    static_assert(nstate==dim+2, "Physics::RealGas() should be created with nstate=dim+2"); // TO DO: UPDATE THIS with nspecies
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

    const real density = primitive_soln[0];
    dealii::Tensor<1,dim,real> vel;;
    for (int d=0; d<dim; ++d) { vel[d] = primitive_soln[1+d]; }
    const real pressure = primitive_soln[nstate-1];
    const real temperature = (this->gam_ref*this->mach_ref_sqr)*pressure/density;
    const real vel2 = compute_velocity_squared(vel);
    const real kinetic_energy = 0.5*density*vel2;
    const real enthalpy = compute_enthalpy(temperature);
    const real specific_internal_energy = enthalpy - (this->R_ref*this->temperature_ref/this->u_ref_sqr)*this->R_Air_NonDim*temperature;
    const real total_energy = density*specific_internal_energy + kinetic_energy; //rhoE
    std::array<real, nstate> conservative_soln;
    conservative_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        conservative_soln[1+d] = density*vel[d];
    }
    conservative_soln[nstate-1] = total_energy;

    return conservative_soln;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 RealGas<dim,nstate,real>
:: compute_density ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = conservative_soln[0];
    return density;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,dim,real2> RealGas<dim,nstate,real>
::compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = compute_density<real2>(conservative_soln);
    dealii::Tensor<1,dim,real2> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}

template <int dim, int nstate, typename real>
template <typename real2>
inline real2 RealGas<dim,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real2> &velocities ) const
{
    real2 vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + velocities[d]*velocities[d]; 
    }    
    
    return vel2;
}

/// It is for NASA polynom1ial
template <int dim, int nstate, typename real>
dealii::Tensor<1,8,real> RealGas<dim,nstate,real>
:: get_NASA_coefficients (const int species) const
{
    dealii::Tensor<1,8,real> NASA_CAP;
    real a1,a2,a3,a4,a5,a6,a7,b1;

    if (species == 1) //Air
    {
        /// It is for Air, T range: 200[K] - 1000[K]
        a1 =  1.009950160e+04;
        a2 = -1.968275610e+02;
        a3 =  5.009155110e+00;
        a4 = -5.761013730e-03;
        a5 =  1.066859930e-05;
        a6 = -7.940297970e-09;
        a7 =  2.185231910e-12;
        b1 = -1.767967310e+02;     
    }
    if (species == 2) // O2: Oxygen, etc...
    {
        //... write data for all the species we use
    }
    NASA_CAP[0] = a1;
    NASA_CAP[1] = a2;
    NASA_CAP[2] = a3;
    NASA_CAP[3] = a4;
    NASA_CAP[4] = a5;
    NASA_CAP[5] = a6;
    NASA_CAP[6] = a7;
    NASA_CAP[7] = b1;

    return NASA_CAP;
}

/// It IS FOR Cp computation
template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
:: compute_Cp ( const real temperature ) const
{
    // This will be changed when you implement multi-species
    int species = 1;

    // NASA_CAP
    dealii::Tensor<1,8,real> NASA_CAP = get_NASA_coefficients(species);
    real a1 = NASA_CAP[0];
    real a2 = NASA_CAP[1];
    real a3 = NASA_CAP[2];
    real a4 = NASA_CAP[3];
    real a5 = NASA_CAP[4];
    real a6 = NASA_CAP[5];
    real a7 = NASA_CAP[6];
    // real b1 = NASA_CAP[7];

    /// dimensinalize... T
    const real T = temperature*this->temperature_ref; // [K]

    /// polynomial
    real Cp = a1/pow(T,2.0) + a2/T + a3 + a4*T + a5*pow(T,2.0) + a6*pow(T,3.0) + a7*pow(T,4.0); // NASA polynomial
    Cp = Cp*this->R_Air_Dim; // Dim
    Cp = Cp/this->R_ref;  // NonDim
    return Cp;
}

/// It is for h computation
template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
:: compute_enthalpy ( const real temperature  ) const
{

    // This will be changed when you implement multi-species
    int species = 1;

    // NASA_CAP
    dealii::Tensor<1,8,real> NASA_CAP = get_NASA_coefficients(species);
    real a1 = NASA_CAP[0];
    real a2 = NASA_CAP[1];
    real a3 = NASA_CAP[2];
    real a4 = NASA_CAP[3];
    real a5 = NASA_CAP[4];
    real a6 = NASA_CAP[5];
    real a7 = NASA_CAP[6];
    real b1 = NASA_CAP[7];

    /// dimensinalize... T
    const real T = temperature*this->temperature_ref; // [K]

    /// polynomial
    real enthalpy = -a1/pow(T,2.0) + a2*(log(T))/T + a3 + a4*T/2 + a5*pow(T,2.0)/3 + a6*pow(T,3.0)/4 + a7*pow(T,4.00)/5 +b1/T; // NASA polynomial
    enthalpy = enthalpy*this->R_Air_Dim*T; // Dim
    enthalpy = enthalpy/this->u_ref_sqr; // NonDim
    return enthalpy;
}

/// IT IS FOR ALGORITHM 3 (Standard Euler for now)
template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
:: compute_temperature ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = compute_density(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real vel2 = compute_velocity_squared(vel);
    const real total_energy = conservative_soln[nstate-1]/density;
    real pressure = (this->gam_ref-1.0)*density*(total_energy - 0.5*vel2);
    real temperature = pressure*(this->gam_ref)*this->mach_ref_sqr / (density);

    return temperature;
}

/// IT IS FOR ALGORITHM 4 (Standard Euler for now)
template <int dim, int nstate, typename real>
template<typename real2>
inline real2 RealGas<dim,nstate,real>
::compute_pressure ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = conservative_soln[0];
    const real temperature = compute_temperature(conservative_soln);
    const real pressure = (density*temperature)/(this->gam_ref*this->mach_ref_sqr);

    return pressure;
}

/// IT IS FOR ALGORITHM 6, IT is only valid for single species. For multi-species, adding algorism 5 and modify largely this.
template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
:: compute_total_enthalpy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = compute_density<real>(conservative_soln);
    const real pressure = compute_pressure<real>(conservative_soln);
    const real total_energy = conservative_soln[3]/density;
    real total_enthalpy = total_energy + pressure/density;

    return total_enthalpy;
}

/// IT IS FOR ALGORITHM 7
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> RealGas<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real density = compute_density<real>(conservative_soln);
    const real pressure = compute_pressure<real>(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    const real total_enthalpy = compute_total_enthalpy(conservative_soln);    //note: missing <real>

    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[nstate-1][flux_dim] = density*vel[flux_dim]*total_enthalpy;
        // TO DO: now loop over nspecies
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    real density = conservative_soln[0];
    const real pressure = compute_pressure<real>(conservative_soln);
    const real sound = sqrt(pressure*this->gam_ref/density);

    return sound;
}

template <int dim, int nstate, typename real>
inline real RealGas<dim,nstate,real>
::compute_kinetic_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    const real vel2 = compute_velocity_squared(vel);
    const real kinetic_energy = 0.50*density*vel2;

    return kinetic_energy;
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

        // Density
          /*computed_quantities(++current_data_index) = primitive_soln[0];*/
            computed_quantities(++current_data_index) = conservative_soln[0];
        // Velocities
        for (unsigned int d=0; d<dim; ++d) {
            /*computed_quantities(++current_data_index) = primitive_soln[1+d];*/
            computed_quantities(++current_data_index) = conservative_soln[1+d]/conservative_soln[0];
        }
        // Momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Energy
        computed_quantities(++current_data_index) = conservative_soln[nstate-1];
        // Pressure
        /*computed_quantities(++current_data_index) = primitive_soln[nstate-1];*/
        computed_quantities(++current_data_index) = compute_pressure<real>(conservative_soln);
        // Pressure coefficient
        /*computed_quantities(++current_data_index) = (primitive_soln[nstate-1] - pressure_inf) / dynamic_pressure_inf;*/
        computed_quantities(++current_data_index) = 999;
        // Temperature
        /*computed_quantities(++current_data_index) = compute_temperature<real>(primitive_soln);*/
        computed_quantities(++current_data_index) = compute_temperature(conservative_soln);     //note: missing <real>
        // Entropy generation
        /*computed_quantities(++current_data_index) = compute_entropy_measure(conservative_soln) - entropy_inf;*/
        computed_quantities(++current_data_index) = 999;
        // Mach Number
        /*computed_quantities(++current_data_index) = compute_mach_number(conservative_soln);*/
        computed_quantities(++current_data_index) = 999;

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
    interpretation.push_back (DCI::component_is_scalar); // Density
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
    }
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Momentum
    }
    interpretation.push_back (DCI::component_is_scalar); // Energy
    interpretation.push_back (DCI::component_is_scalar); // Pressure
    interpretation.push_back (DCI::component_is_scalar); // Pressure coefficient
    interpretation.push_back (DCI::component_is_scalar); // Temperature
    interpretation.push_back (DCI::component_is_scalar); // Entropy generation
    interpretation.push_back (DCI::component_is_scalar); // Mach number

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
    names.push_back ("density");
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("velocity");
    }
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("momentum");
    }
    names.push_back ("energy");
    names.push_back ("pressure");
    names.push_back ("pressure_coeffcient");
    names.push_back ("temperature");

    names.push_back ("entropy_generation");
    names.push_back ("mach_number");
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
template class RealGas < PHILIP_DIM, PHILIP_DIM+2, double     >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2, FadType    >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2, RadType    >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RealGas < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace