#include <cmath>
#include <vector>
#include <complex> // for the jacobian
#include <deal.II/lac/identity_matrix.h>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
NavierStokes<dim, nstate, real>::NavierStokes( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : Euler<dim,nstate,real>(ref_length, 
                             gamma_gas, 
                             mach_inf, 
                             angle_of_attack, 
                             side_slip_angle, 
                             manufactured_solution_function,
                             two_point_num_flux_type,
                             true,  //has_nonzero_diffusion = true
                             false) //has_nonzero_physical_source = false
    , viscosity_coefficient_inf(1.0) // Nondimensional - Free stream values
    , use_constant_viscosity(use_constant_viscosity)
    , constant_viscosity(constant_viscosity) // Nondimensional - Free stream values
    , prandtl_number(prandtl_number)
    , reynolds_number_inf(reynolds_number_inf)
    , isothermal_wall_temperature(isothermal_wall_temperature) // Nondimensional - Free stream values
    , thermal_boundary_condition_type(thermal_boundary_condition_type)
    , sutherlands_temperature(110.4) // Sutherland's temperature. Units: [K]
    , freestream_temperature(temperature_inf) // Freestream temperature. Units: [K]
    , temperature_ratio(sutherlands_temperature/freestream_temperature)
{
    static_assert(nstate==dim+2, "Physics::NavierStokes() should be created with nstate=dim+2");
    // Nothing to do here so far
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> NavierStokes<dim,nstate,real>
::compute_temperature_gradient (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    const real2 density = primitive_soln[0];
    const real2 temperature = this->template compute_temperature<real2>(primitive_soln); // from Euler

    dealii::Tensor<1,dim,real2> temperature_gradient;
    for (int d=0; d<dim; d++) {
        temperature_gradient[d] = (this->gam*this->mach_inf_sqr*primitive_soln_gradient[nstate-1][d] - temperature*primitive_soln_gradient[0][d])/density;
    }
    return temperature_gradient;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> NavierStokes<dim,nstate,real>
::compute_velocities_parallel_to_wall(
    const std::array<real2,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real2> &normal_vector) const
{
    // extract velocities
    const dealii::Tensor<1,dim,real2> velocities = this->template compute_velocities<real2>(conservative_soln);// from Euler
    // compute normal velocity
    real2 normal_velocity = 0.0;
    for(int d=0;d<dim;++d){
        normal_velocity += velocities[d]*normal_vector[d];
    }
    // compute wall parallel velocities
    dealii::Tensor<1,dim,real2> velocities_parallel_to_wall;
    for(int d=0;d<dim;++d){
        velocities_parallel_to_wall[d] = velocities[d] - normal_velocity*normal_vector[d];
    }
    return velocities_parallel_to_wall;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> NavierStokes<dim,nstate,real>
::compute_wall_tangent_vector(
    const std::array<real2,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real2> &normal_vector) const
{
    // compute wall parallel velocities
    const dealii::Tensor<1,dim,real2> velocities_parallel_to_wall = compute_velocities_parallel_to_wall(conservative_soln,normal_vector);
    // get magnitude
    real2 magnitude = 0.0;
    for(int d=0;d<dim;++d){
        magnitude += velocities_parallel_to_wall[d]*velocities_parallel_to_wall[d];
    }
    magnitude = pow(magnitude,0.5);
    // compute tangent vector
    dealii::Tensor<1,dim,real2> tangent_vector;
    for(int d=0;d<dim;++d){
        tangent_vector[d] = velocities_parallel_to_wall[d]/magnitude;
    }
    return tangent_vector;
}

template <int dim, int nstate, typename real>
template<typename real2>
real2 NavierStokes<dim,nstate,real>
::compute_wall_shear_stress (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient,
    const dealii::Tensor<1,dim,real2> &normal_vector) const
{
    // Computes the non-dimensional wall shear stress
    // - get primitive solution, gradient, and velocities gradient 
    const std::array<real2,nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real2>(conservative_soln); // from Euler
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient
                 = this->template convert_conservative_gradient_to_primitive_gradient_templated<real2>(conservative_soln,conservative_soln_gradient);
    
    // const dealii::Tensor<2,dim,real2> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    /*
    // - compute normal velocity gradient
    dealii::Tensor<1,dim,real2> velocity_normal_to_wall_gradient;
    for(int dspace=0;dspace<dim;++dspace){
        velocity_normal_to_wall_gradient[dspace] = 0.0;
        for(int dvel=0;dvel<dim;++dvel){
            velocity_normal_to_wall_gradient[dspace] += velocities_gradient[dvel][dspace]*normal_vector[dvel];
        }
    }
    // - compute wall parallel velocities gradient
    dealii::Tensor<2,dim,real2> velocities_parallel_to_wall_gradient;
    for(int dspace=0;dspace<dim;++dspace){
        for(int dvel=0;dvel<dim;++dvel){
            velocities_parallel_to_wall_gradient[dvel][dspace] = velocities_gradient[dvel][dspace] - velocity_normal_to_wall_gradient[dspace]*normal_vector[dvel];
        }
    }
    // - compute wall parallel velocity gradient
    dealii::Tensor<1,dim,real2> velocity_parallel_to_wall_gradient;
    for(int dspace=0;dspace<dim;++dspace){
        real2 magnitude = 0.0;
        for(int dvel=0;dvel<dim;++dvel){
            magnitude += velocities_parallel_to_wall_gradient[dvel][dspace]*velocities_parallel_to_wall_gradient[dvel][dspace];
        }
        velocity_parallel_to_wall_gradient[dspace] = pow(magnitude,0.5);
    }
    */
    /*
    // - compute wall parallel velocity gradient in the direction normal to the wall
    real2 velocity_gradient_of_parallel_velocity_in_the_direction_normal_to_wall = 0.0;
    for(int d=0;d<dim;++d){
        velocity_gradient_of_parallel_velocity_in_the_direction_normal_to_wall += velocities_gradient[0][d]*normal_vector[d]; // for channel flow (simplest case, x-velocity is the wall parallel velocity)
        // velocity_gradient_of_parallel_velocity_in_the_direction_normal_to_wall += velocity_parallel_to_wall_gradient[d]*normal_vector[d];
    }
    // Reference: https://www.cfd-online.com/Wiki/Wall_shear_stress
    const real2 scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient<real2>(primitive_soln);
    const real2 wall_shear_stress = scaled_viscosity_coefficient*velocity_gradient_of_parallel_velocity_in_the_direction_normal_to_wall;
    */
    // For 3D flow over curved walls, reference: https://www.cfd-online.com/Forums/main/11103-calculate-y-u-how-get-wall-shear-stress.html
    // const dealii::Tensor<1,dim,real2> tangent_vector = compute_wall_tangent_vector<real2>(conservative_soln,normal_vector);
    const dealii::Tensor<2,dim,real2> viscous_stress_tensor = compute_viscous_stress_tensor<real2>(primitive_soln,primitive_soln_gradient);
    // real2 wall_shear_stress = 0.0;
    // for(int i=0;i<dim;++i){
    //     real2 val = 0.0;
    //     for(int j=0;j<dim;++j){
    //         val += viscous_stress_tensor[i][j]*tangent_vector[j];
    //     }
    //     wall_shear_stress += val*val;
    // }
    // wall_shear_stress = pow(wall_shear_stress,0.5);

    /*// build tangential operator (can be used to get the surface tangential component of some vector)
    dealii::Tensor<2,dim,real2> tangential_operator;
    for(int i=0;i<dim;++i){
        for(int j=0;j<dim;++j){
            tangential_operator[i][j] = 0.0; // initialize
            if(j==i) tangential_operator[i][j] = 1.0;
            tangential_operator[i][j] -= normal_vector[j]*normal_vector[i];
        }
    }*/
    // viscous stress tensor times normal vector (contains all components on the surface associated with the normal vector)
    dealii::Tensor<1,dim,real2> viscous_stress_tensor_times_normal_vector;
    for(int i=0;i<dim;++i){
        viscous_stress_tensor_times_normal_vector[i] = 0.0;
        for(int j=0;j<dim;++j){
            viscous_stress_tensor_times_normal_vector[i] += viscous_stress_tensor[i][j]*normal_vector[j];
        }
    }
    /*// components tangent to the surface
    dealii::Tensor<1,dim,real2> viscous_stress_tensor_times_tangent_vector;
    for(int i=0;i<dim;++i){
        viscous_stress_tensor_times_tangent_vector[i] = 0.0;
        for(int j=0;j<dim;++j){
            viscous_stress_tensor_times_tangent_vector[i] += tangential_operator[i][j]*viscous_stress_tensor_times_normal_vector[j];
        }
    }*/
    // compute magnitude
    real2 wall_shear_stress = 0.0;
    for(int i=0;i<dim;++i){
        // wall_shear_stress += viscous_stress_tensor_times_tangent_vector[i]*viscous_stress_tensor_times_tangent_vector[i];
        wall_shear_stress += viscous_stress_tensor_times_normal_vector[i]*viscous_stress_tensor_times_normal_vector[i];
    }
    wall_shear_stress = pow(wall_shear_stress,0.5);


    return wall_shear_stress;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const
{   
    // Use either Sutherland's law or constant viscosity
    real2 viscosity_coefficient;
    if(use_constant_viscosity){
        viscosity_coefficient = 1.0*constant_viscosity;
    } else {
        viscosity_coefficient = compute_viscosity_coefficient_sutherlands_law<real2>(primitive_soln);
    }

    return viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient_from_temperature (const real2 temperature) const
{   
    // Use either Sutherland's law or constant viscosity
    real2 viscosity_coefficient;
    if(use_constant_viscosity){
        viscosity_coefficient = 1.0*constant_viscosity;
    } else {
        viscosity_coefficient = compute_viscosity_coefficient_sutherlands_law_from_temperature<real2>(temperature);
    }

    return viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient_sutherlands_law (const std::array<real2,nstate> &primitive_soln) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
     
    const real2 temperature = this->template compute_temperature<real2>(primitive_soln); // from Euler

    const real2 viscosity_coefficient = compute_viscosity_coefficient_sutherlands_law_from_temperature<real2>(temperature);
    
    return viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient_sutherlands_law_from_temperature (const real2 temperature) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const real2 viscosity_coefficient = ((1.0 + temperature_ratio)/(temperature + temperature_ratio))*pow(temperature,1.5);
    
    return viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::scale_viscosity_coefficient (const real2 viscosity_coefficient) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const real2 scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;
    
    return scaled_viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_scaled_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const real2 viscosity_coefficient = compute_viscosity_coefficient<real2>(primitive_soln);
    const real2 scaled_viscosity_coefficient = scale_viscosity_coefficient(viscosity_coefficient);

    return scaled_viscosity_coefficient;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number (const real2 scaled_viscosity_coefficient, const double prandtl_number_input) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$, given the scaled viscosity coefficient
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const real2 scaled_heat_conductivity = scaled_viscosity_coefficient/(this->gamm1*this->mach_inf_sqr*prandtl_number_input);
    
    return scaled_heat_conductivity;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 NavierStokes<dim,nstate,real>
::compute_scaled_heat_conductivity (const std::array<real2,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const real2 scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient<real2>(primitive_soln);

    const real2 scaled_heat_conductivity = compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_viscosity_coefficient,prandtl_number);
    
    return scaled_heat_conductivity;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> NavierStokes<dim,nstate,real>
::compute_heat_flux (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const real2 scaled_heat_conductivity = compute_scaled_heat_conductivity<real2>(primitive_soln);
    const dealii::Tensor<1,dim,real2> temperature_gradient = compute_temperature_gradient<real2>(primitive_soln, primitive_soln_gradient);
    // Compute the heat flux
    const dealii::Tensor<1,dim,real2> heat_flux = compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<real2>(scaled_heat_conductivity,temperature_gradient);
    return heat_flux;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> NavierStokes<dim,nstate,real>
::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient (
    const real2 scaled_heat_conductivity,
    const dealii::Tensor<1,dim,real2> &temperature_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,real2> heat_flux;
    for (int d=0; d<dim; d++) {
        heat_flux[d] = -scaled_heat_conductivity*temperature_gradient[d];
    }
    return heat_flux;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,3,real2> NavierStokes<dim,nstate,real>
::compute_vorticity (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const
{
    // Compute the vorticity
    dealii::Tensor<1,3,real2> vorticity;
    for(int d=0; d<3; ++d) {
        vorticity[d] = 0.0;
    }
    if constexpr(dim>1) {
        // Get velocity gradient
        const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real2>(conservative_soln, conservative_soln_gradient);
        const dealii::Tensor<2,dim,real2> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
        if constexpr(dim==2) {
            // vorticity exists only in z-component
            vorticity[2] = velocities_gradient[1][0] - velocities_gradient[0][1]; // z-component
        }
        if constexpr(dim==3) {
            vorticity[0] = velocities_gradient[2][1] - velocities_gradient[1][2]; // x-component
            vorticity[1] = velocities_gradient[0][2] - velocities_gradient[2][0]; // y-component
            vorticity[2] = velocities_gradient[1][0] - velocities_gradient[0][1]; // z-component
        }
    }
    return vorticity;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_vorticity_magnitude_sqr (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute the vorticity
    dealii::Tensor<1,3,real> vorticity = compute_vorticity(conservative_soln, conservative_soln_gradient);
    // Compute vorticity magnitude squared
    real vorticity_magnitude_sqr = 0.0;
    for(int d=0; d<3; ++d) {
        vorticity_magnitude_sqr += vorticity[d]*vorticity[d];
    }
    return vorticity_magnitude_sqr;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_vorticity_magnitude (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    real vorticity_magnitude_sqr = compute_vorticity_magnitude_sqr(conservative_soln, conservative_soln_gradient);
    real vorticity_magnitude = sqrt(vorticity_magnitude_sqr); 
    return vorticity_magnitude;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_enstrophy (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute enstrophy
    const real density = conservative_soln[0];
    real enstrophy = 0.5*density*compute_vorticity_magnitude_sqr(conservative_soln, conservative_soln_gradient);
    return enstrophy;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_vorticity_based_dissipation_rate_from_integrated_enstrophy (
    const real integrated_enstrophy) const
{
    real dissipation_rate = 2.0*integrated_enstrophy/(this->reynolds_number_inf);
    return dissipation_rate;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_pressure_dilatation (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get pressure
    const real pressure = this->template compute_pressure<real>(conservative_soln);

    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real>(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real>(primitive_soln_gradient);

    // Compute the pressure dilatation
    real pressure_dilatation = 0.0;
    for(int d=0; d<dim; ++d) {
        pressure_dilatation += velocities_gradient[d][d]; // divergence
    }
    pressure_dilatation *= pressure;

    return pressure_dilatation;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes<dim,nstate,real>
::compute_strain_rate_tensor_from_conservative (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    return compute_strain_rate_tensor_from_conservative_templated<real>(conservative_soln,conservative_soln_gradient);
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::compute_strain_rate_tensor_from_conservative_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const
{
    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real2>(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real2> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);

    // Strain rate tensor, S_{i,j}
    const dealii::Tensor<2,dim,real2> strain_rate_tensor = compute_strain_rate_tensor<real2>(velocities_gradient);
    return strain_rate_tensor;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes<dim,nstate,real>
::compute_deviatoric_strain_rate_tensor (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real>(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real>(primitive_soln_gradient);

    // Strain rate tensor, S_{i,j}
    const dealii::Tensor<2,dim,real> strain_rate_tensor = compute_strain_rate_tensor<real>(velocities_gradient);
    
    // Compute divergence of velocity
    real vel_divergence = 0.0;
    for(int d1=0; d1<dim; ++d1) {
        vel_divergence += velocities_gradient[d1][d1];
    }

    // Compute the deviatoric strain rate tensor
    dealii::Tensor<2,dim,real> deviatoric_strain_rate_tensor;
    for(int d1=0; d1<dim; ++d1) {
        for(int d2=0; d2<dim; ++d2) {
            deviatoric_strain_rate_tensor[d1][d2] = strain_rate_tensor[d1][d2] - (1.0/3.0)*vel_divergence;
        }
    }
    return deviatoric_strain_rate_tensor;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::get_tensor_magnitude_sqr (
    const dealii::Tensor<2,dim,real> &tensor) const
{
    real tensor_magnitude_sqr = 0.0;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            tensor_magnitude_sqr += tensor[i][j]*tensor[i][j];
        }
    }
    return tensor_magnitude_sqr;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::get_tensor_magnitude (
    const dealii::Tensor<2,dim,real> &tensor) const
{
    return sqrt(get_tensor_magnitude_sqr(tensor));
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_deviatoric_strain_rate_tensor_magnitude_sqr (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute the deviatoric strain rate tensor
    const dealii::Tensor<2,dim,real> deviatoric_strain_rate_tensor = compute_deviatoric_strain_rate_tensor(conservative_soln,conservative_soln_gradient);
    // Get magnitude squared
    real deviatoric_strain_rate_tensor_magnitude_sqr = get_tensor_magnitude_sqr(deviatoric_strain_rate_tensor);
    
    return deviatoric_strain_rate_tensor_magnitude_sqr;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr (
    const real integrated_deviatoric_strain_rate_tensor_magnitude_sqr) const
{
    real dissipation_rate = 2.0*integrated_deviatoric_strain_rate_tensor_magnitude_sqr/(this->reynolds_number_inf);
    return dissipation_rate;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::extract_velocities_gradient_from_primitive_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    dealii::Tensor<2,dim,real2> velocities_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            velocities_gradient[d1][d2] = primitive_soln_gradient[1+d1][d2];
        }
    }
    return velocities_gradient;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::compute_strain_rate_tensor (
    const dealii::Tensor<2,dim,real2> &vel_gradient) const
{ 
    // Strain rate tensor, S_{i,j}
    dealii::Tensor<2,dim,real2> strain_rate_tensor;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            // rate of strain (deformation) tensor:
            strain_rate_tensor[d1][d2] = 0.5*(vel_gradient[d1][d2] + vel_gradient[d2][d1]);
        }
    }
    return strain_rate_tensor;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_strain_rate_tensor_magnitude_sqr (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real>(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real>(primitive_soln_gradient);

    // Compute the strain rate tensor
    const dealii::Tensor<2,dim,real> strain_rate_tensor = compute_strain_rate_tensor(velocities_gradient);
    // Get magnitude squared
    real strain_rate_tensor_magnitude_sqr = get_tensor_magnitude_sqr(strain_rate_tensor);
    
    return strain_rate_tensor_magnitude_sqr;
}

template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::compute_strain_rate_tensor_based_dissipation_rate_from_integrated_strain_rate_tensor_magnitude_sqr (
    const real integrated_strain_rate_tensor_magnitude_sqr) const
{
    real dissipation_rate = 2.0*integrated_strain_rate_tensor_magnitude_sqr/(this->reynolds_number_inf);
    return dissipation_rate;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor (
    const real2 scaled_viscosity_coefficient,
    const dealii::Tensor<2,dim,real2> &strain_rate_tensor) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */

    // Divergence of velocity
    // -- Initialize
    real2 vel_divergence; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){ 
        vel_divergence = 0.0;
    }
    // -- Obtain from trace of strain rate tensor
    for (int d=0; d<dim; d++) {
        vel_divergence += strain_rate_tensor[d][d];
    }

    // Viscous stress tensor, \tau_{i,j}
    dealii::Tensor<2,dim,real2> viscous_stress_tensor;
    const real2 scaled_2nd_viscosity_coefficient = (-2.0/3.0)*scaled_viscosity_coefficient; // Stokes' hypothesis
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            viscous_stress_tensor[d1][d2] = 2.0*scaled_viscosity_coefficient*strain_rate_tensor[d1][d2];
        }
        viscous_stress_tensor[d1][d1] += scaled_2nd_viscosity_coefficient*vel_divergence;
    }
    return viscous_stress_tensor;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::compute_viscous_stress_tensor_from_conservative_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const
{
    // Step 1: Primitive solution
    const std::array<real2,nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real2>(conservative_soln); // from Euler
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real2>(conservative_soln, conservative_soln_gradient);

    // Viscous stress tensor, \tau_{i,j}
    const dealii::Tensor<2,dim,real2> viscous_stress_tensor = compute_viscous_stress_tensor<real2>(primitive_soln,primitive_soln_gradient);

    return viscous_stress_tensor;
}

template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> NavierStokes<dim,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    const dealii::Tensor<2,dim,real2> vel_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const dealii::Tensor<2,dim,real2> strain_rate_tensor = compute_strain_rate_tensor<real2>(vel_gradient);
    const real2 scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient<real2>(primitive_soln);

    // Viscous stress tensor, \tau_{i,j}
    const dealii::Tensor<2,dim,real2> viscous_stress_tensor 
        = compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<real2>(scaled_viscosity_coefficient,strain_rate_tensor);

    return viscous_stress_tensor;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes<dim,nstate,real>
::compute_germano_idendity_matrix_L_component (
    const std::array<real,nstate> &conservative_soln) const
{
    const dealii::Tensor<1,dim,real> vel = this->template compute_velocities<real>(conservative_soln);
    dealii::Tensor<2,dim,real> matrix_L;
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            matrix_L[i][j] = vel[i]*vel[j];
        }
    }
    return matrix_L;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes<dim,nstate,real>
::compute_germano_idendity_matrix_M_component (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    dealii::Tensor<2,dim,real> matrix_M;

    // Strain rate tensor, S_{i,j}
    const dealii::Tensor<2,dim,real> strain_rate_tensor = compute_strain_rate_tensor_from_conservative(conservative_soln, conservative_soln_gradient);
    const real strain_rate_tensor_magnitude = get_tensor_magnitude(strain_rate_tensor);

    // Compute divergence of velocity
    real strain_rate_tensor_trace = 0.0;
    for(int i=0; i<dim; ++i) {
        strain_rate_tensor_trace += strain_rate_tensor[i][i];
    }

    // Compute the deviatoric strain rate tensor
    dealii::Tensor<2,dim,real> deviatoric_strain_rate_tensor;
    for(int i=0; i<dim; ++i) {
        for(int j=0; j<dim; ++j) {
            matrix_M[i][j] = strain_rate_tensor_magnitude*strain_rate_tensor[i][j];
        }
        matrix_M[i][i] -= strain_rate_tensor_magnitude*(1.0/3.0)*strain_rate_tensor_trace;
    }
    return matrix_M;
}

//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real NavierStokes<dim,nstate,real>
::get_tensor_product_magnitude_sqr (
    const dealii::Tensor<2,dim,real> &tensor1,
    const dealii::Tensor<2,dim,real> &tensor2) const
{
    real tensor_product_magnitude_sqr = 0.0;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            tensor_product_magnitude_sqr += tensor1[i][j]*tensor2[i][j];
        }
    }
    return tensor_product_magnitude_sqr;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux = dissipative_flux_templated<real>(conservative_soln, solution_gradient);
    return viscous_flux;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes<dim,nstate,real>
::compute_scaled_viscosity_gradient (
    const std::array<real,nstate> &primitive_soln,
    const dealii::Tensor<1,dim,real> &temperature_gradient) const
{
    /* Gradient of the scaled nondimensionalized viscosity coefficient
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14 and 4.14.17)
     */
    const real temperature = this->compute_temperature(primitive_soln); // from Euler
    const real scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    // Eq.(4.14.17)
    real dmudT = 0.5*(scaled_viscosity_coefficient/(temperature + temperature_ratio))*(1.0 + 3.0*temperature_ratio/temperature);

    // Gradient (dmudX) from dmudT and dTdX
    dealii::Tensor<1,dim,real> scaled_viscosity_coefficient_gradient;
    for (int d=0; d<dim; d++) {
        scaled_viscosity_coefficient_gradient[d] = dmudT*temperature_gradient[d];
    }

    return scaled_viscosity_coefficient_gradient;
}

// Returns the value from a CoDiPack or Sacado variable.
template<typename real>
double getValue(const real &x) {
    if constexpr(std::is_same<real,double>::value) {
        return x;
    }
    else if constexpr(std::is_same<real,FadType>::value) {
        return x.val(); // sacado
    } 
    else if constexpr(std::is_same<real,FadFadType>::value) {
        return x.val().val(); // sacado
    }
    else if constexpr(std::is_same<real,RadType>::value) {
      return x.value(); // CoDiPack
    } 
    else if(std::is_same<real,RadFadType>::value) {
        return x.value().value(); // CoDiPack
    }
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> NavierStokes<dim,nstate,real>
::dissipative_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
        for (int d=0;d<dim;d++) {
            AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> NavierStokes<dim,nstate,real>
::dissipative_flux_directional_jacobian_wrt_gradient_component (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal,
    const int d_gradient) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; s++) {
        AD_conservative_soln[s] = getValue<real>(conservative_soln[s]);
        for (int d=0;d<dim;d++) {
            if(d == d_gradient){
                adtype ADvar(nstate, s, getValue<real>(solution_gradient[s][d])); // create AD variable
                AD_solution_gradient[s][d] = ADvar;
            }
            else {
                AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
            }
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokes<dim,nstate,real>
::dissipative_source_term (
    const dealii::Point<dim,real> &pos) const
{    
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = this->get_manufactured_solution_value(pos); // from Euler
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = this->get_manufactured_solution_gradient(pos); // from Euler
    
    // Get Manufactured Solution hessian
    std::array<dealii::SymmetricTensor<2,dim,real>,nstate> manufactured_solution_hessian;
    for (int s=0; s<nstate; s++) {
        dealii::SymmetricTensor<2,dim,real> hessian = this->manufactured_solution_function->hessian(pos,s);
        for (int dr=0;dr<dim;dr++) {
            for (int dc=0;dc<dim;dc++) {
                manufactured_solution_hessian[s][dr][dc] = hessian[dr][dc];
            }
        }
    }

    // First term -- wrt to the conservative variables
    // This is similar, should simply provide this function a flux_directional_jacobian() -- could restructure later
    dealii::Tensor<1,nstate,real> dissipative_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = dissipative_flux_directional_jacobian(manufactured_solution, manufactured_solution_gradient, normal);
        
        // get the directional jacobian wrt gradient
        std::array<dealii::Tensor<2,nstate,real>,dim> jacobian_wrt_gradient;
        for (int d_gradient=0;d_gradient<dim;d_gradient++) {
            
            // get the directional jacobian wrt gradient component (x,y,z)
            const dealii::Tensor<2,nstate,real> jacobian_wrt_gradient_component = dissipative_flux_directional_jacobian_wrt_gradient_component(manufactured_solution, manufactured_solution_gradient, normal, d_gradient);
            
            // store each component in jacobian_wrt_gradient -- could do this in the function used above
            for (int sr = 0; sr < nstate; ++sr) {
                for (int sc = 0; sc < nstate; ++sc) {
                    jacobian_wrt_gradient[d_gradient][sr][sc] = jacobian_wrt_gradient_component[sr][sc];
                }
            }
        }
        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d];
                // Second term -- wrt to the gradient of conservative variables
                // -- add the contribution of each gradient component (e.g. x,y,z for dim==3)
                for (int d_gradient=0;d_gradient<dim;d_gradient++) {
                    jac_grad_row += jacobian_wrt_gradient[d_gradient][sr][sc]*manufactured_solution_hessian[sc][d_gradient][d]; // symmetric so d indexing works both ways
                }
            }
            dissipative_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> dissipative_source_term;
    for (int s=0; s<nstate; s++) {
        dissipative_source_term[s] = dissipative_flux_divergence[s];
    }

    return dissipative_source_term;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokes<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/) const
{
    // will probably have to change this line: -- modify so we only need to provide a jacobian
    const std::array<real,nstate> conv_source_term = this->convective_source_term(pos);
    const std::array<real,nstate> diss_source_term = dissipative_source_term(pos);
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++)
    {
        source_term[s] = conv_source_term[s] + diss_source_term[s];
    }
    return source_term;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> NavierStokes<dim,nstate,real>
::convective_flux_directional_jacobian_via_dfad (
    std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
    }

    // Compute AD convective flux
    // -- taken exactly from euler.cpp:
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_conv_flux;
    const adtype density = AD_conservative_soln[0];
    const adtype pressure = this->template compute_pressure<adtype>(AD_conservative_soln);
    const dealii::Tensor<1,dim,adtype> vel = this->template compute_velocities<adtype>(AD_conservative_soln);
    const adtype specific_total_energy = AD_conservative_soln[nstate-1]/AD_conservative_soln[0];
    const adtype specific_total_enthalpy = specific_total_energy + pressure/density;
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        AD_conv_flux[0][flux_dim] = AD_conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            AD_conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        AD_conv_flux[1+flux_dim][flux_dim] += pressure; // Add diagonal of pressure
        // Energy equation
        AD_conv_flux[nstate-1][flux_dim] = density*vel[flux_dim]*specific_total_enthalpy;
    }
    // -- end of computing the AD convective flux

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_conv_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}

template <int dim, int nstate, typename real>
inline real NavierStokes<dim,nstate,real>
::compute_scaled_viscosity_coefficient_derivative_wrt_temperature_via_dfad (
    std::array<real,nstate> &conservative_soln) const
{
    using adtype = FadType;

    // Step 1: Primitive solution
    const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real>(conservative_soln); // from Euler
    
    // Step 2: Compute temperature
    real temperature = this->template compute_temperature<real>(primitive_soln); // from Euler

    // Initialize AD objects
    adtype AD_temperature(1, 0, getValue<real>(temperature));
    
    // Compute the AD scaled viscosity coefficient
    adtype viscosity_coefficient = ((1.0 + temperature_ratio)/(AD_temperature + temperature_ratio))*pow(AD_temperature,1.5);
    adtype scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;

    // Get the derivative from AD
    real dmudT = scaled_viscosity_coefficient.dx(0);

    return dmudT;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> NavierStokes<dim,nstate,real>
::dissipative_flux_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    // Step 1: Primitive solution
    const std::array<real2,nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real2>(conservative_soln); // from Euler
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real2>(conservative_soln, solution_gradient);
    
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<2,dim,real2> viscous_stress_tensor = compute_viscous_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real2> vel = this->template extract_velocities_from_primitive<real2>(primitive_soln); // from Euler
    const dealii::Tensor<1,dim,real2> heat_flux = compute_heat_flux<real2>(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    const std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux = dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<real2>(vel,viscous_stress_tensor,heat_flux);
    return viscous_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokes<dim,nstate,real>
::dissipative_flux_dot_normal (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &/*filtered_solution*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*filtered_solution_gradient*/,
        const bool on_boundary,
        const dealii::types::global_dof_index /*cell_index*/,
        const dealii::Tensor<1,dim,real> &normal,
        const int boundary_type)
{
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux;
    std::array<real,nstate> dissipative_flux_dot_normal;
    dissipative_flux_dot_normal.fill(0.0); // initialize
    // Associated thermal boundary condition
    if((on_boundary && (thermal_boundary_condition_type == thermal_boundary_condition_enum::adiabatic))
        && ((boundary_type == 1001) || (boundary_type == 1006))) { 

        /** If adiabatic on either slip (1001) or no-slip (1006) wall BCs */
        // adiabatic boundary
        // --> Modify viscous flux such that normal_vector dot gradient of temperature must be zero

        // REFERENCES:
        /* (1) Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
         * (2) For the boundary condition case, refer to the equation above equation 458 of the following paper:
         *  Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
         */

        // Step 1: Primitive solution
        const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real>(solution); // from Euler
        
        // Step 2: Gradient of primitive solution
        const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->template convert_conservative_gradient_to_primitive_gradient_templated<real>(solution, solution_gradient);
        
        // Step 3: Viscous stress tensor, Velocities, Heat flux
        const dealii::Tensor<2,dim,real> viscous_stress_tensor = compute_viscous_stress_tensor<real>(primitive_soln, primitive_soln_gradient);
        const dealii::Tensor<1,dim,real> vel = this->template extract_velocities_from_primitive<real>(primitive_soln); // from Euler
        /* ---> Impose adiabatic boundary condition by modifying the heat flux. */
        dealii::Tensor<1,dim,real> heat_flux;
        for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
            // set the heat flux to zero since we want the normal dot gradient of temperature to be zero for an adiabatic boundary
            heat_flux[flux_dim] = 0.0;
        }

        // Step 4: Construct viscous flux; Note: sign corresponds to LHS
        dissipative_flux = dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<real>(vel,viscous_stress_tensor,heat_flux);
    } else {
        // if not on boundary and for all other types of boundary conditions (including isothermal) --> no change to dissipative flux
        // no change to dissipative flux for BCs that do not impose a condition on the gradient at the boundary
        dissipative_flux = dissipative_flux_templated<real>(solution,solution_gradient);
    }

    // compute the dot product with the normal vector
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<dim; ++d) {
            dissipative_flux_dot_normal[s] += dissipative_flux[s][d] * normal[d];//compute dot product
        }
    }

    return dissipative_flux_dot_normal;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> NavierStokes<dim,nstate,real>
::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux (
    const dealii::Tensor<1,dim,real2> &vel,
    const dealii::Tensor<2,dim,real2> &viscous_stress_tensor,
    const dealii::Tensor<1,dim,real2> &heat_flux) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    /* Construct viscous flux given velocities, viscous stress tensor,
     * and heat flux; Note: sign corresponds to LHS
     */
    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux;
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        viscous_flux[0][flux_dim] = 0.0;
        // Momentum equation
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
            viscous_flux[1+stress_dim][flux_dim] = -viscous_stress_tensor[stress_dim][flux_dim];
        }
        // Energy equation
        viscous_flux[nstate-1][flux_dim] = 0.0;
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
           viscous_flux[nstate-1][flux_dim] -= vel[stress_dim]*viscous_stress_tensor[flux_dim][stress_dim];
        }
        viscous_flux[nstate-1][flux_dim] += heat_flux[flux_dim];
    }
    return viscous_flux;
}

template <int dim, int nstate, typename real>
void NavierStokes<dim,nstate,real>
::boundary_face_values_viscous_flux (
        const int boundary_type,
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        const std::array<real,nstate> &/*filtered_soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*filtered_soln_grad_int*/,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1000) {
        // Manufactured solution boundary condition
        boundary_manufactured_solution (pos, normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1001) {
        // Wall boundary condition
        boundary_wall_viscous_flux (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    }
}

template <int dim, int nstate, typename real>
void NavierStokes<dim,nstate,real>
::boundary_wall_viscous_flux (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;

    // No-slip wall boundary conditions

    // Apply boundary conditions:
    // -- solution at boundary
    soln_bc[0] = soln_int[0];
    soln_bc[nstate-1] = soln_int[nstate-1];
    for (int d=0; d<dim; ++d) {
        soln_bc[1+d] = -soln_int[1+d];
    }
    // -- gradient of solution at boundary
    for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
    // If adiabatic wall, set gradient to zero
    if(thermal_boundary_condition_type == thermal_boundary_condition_enum::adiabatic){
        soln_grad_bc[nstate-1] = 0.0;
    }
}

template <int dim, int nstate, typename real>
void NavierStokes<dim,nstate,real>
::boundary_manufactured_solution (
    const dealii::Point<dim, real> &pos,
    const dealii::Tensor<1,dim,real> &/*normal_int*/,
    const std::array<real,nstate> &/*soln_int*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
    std::array<real,nstate> &soln_bc,
    std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Manufactured solution boundary condition 
    // Note: This is consistent with Navah & Nadarajah (2018)
    std::array<real,nstate> boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
    for (int i=0; i<nstate; i++) {
        boundary_values[i] = this->manufactured_solution_function->value (pos, i);
        boundary_gradients[i] = this->manufactured_solution_function->gradient (pos, i);
    }
    for (int istate=0; istate<nstate; istate++) {
        soln_bc[istate] = boundary_values[istate];
        // soln_grad_bc[istate] = soln_grad_int[istate]; // done in convection_diffusion.cpp
        soln_grad_bc[istate] = boundary_gradients[istate];
    }
}

template <int dim, int nstate, typename real>
dealii::Vector<double> NavierStokes<dim,nstate,real>::post_compute_derived_quantities_vector (
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
        const std::array<double, nstate> primitive_soln = this->template convert_conservative_to_primitive_templated<real>(conservative_soln);
        // if (primitive_soln[0] < 0) this->pcout << evaluation_points << std::endl;

        std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient;
        for (unsigned int s=0; s<nstate; ++s) {
            for (unsigned int d=0; d<dim; ++d) {
                conservative_soln_gradient[s][d] = duh[s][d];
            }
        }

        // Density
        computed_quantities(++current_data_index) = primitive_soln[0];
        // Velocities
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = primitive_soln[1+d];
        }
        // Momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Total Energy
        computed_quantities(++current_data_index) = conservative_soln[nstate-1];
        // Pressure
        computed_quantities(++current_data_index) = primitive_soln[nstate-1];
        // Pressure coefficient
        computed_quantities(++current_data_index) = (primitive_soln[nstate-1] - this->pressure_inf) / this->dynamic_pressure_inf;
        // Temperature
        computed_quantities(++current_data_index) = this->template compute_temperature<real>(primitive_soln);
        // Entropy generation
        computed_quantities(++current_data_index) = this->compute_entropy_measure(conservative_soln) - this->entropy_inf;
        // Mach Number
        computed_quantities(++current_data_index) = this->compute_mach_number(conservative_soln);
        if constexpr(dim==3) {
            // Vorticity
            dealii::Tensor<1,3,double> vorticity = compute_vorticity<double>(conservative_soln,conservative_soln_gradient);
            for (unsigned int d=0; d<3; ++d) {
                computed_quantities(++current_data_index) = vorticity[d];
            }
        }
        // Vorticity magnitude
        computed_quantities(++current_data_index) = compute_vorticity_magnitude(conservative_soln,conservative_soln_gradient);
        // Enstrophy
        computed_quantities(++current_data_index) = compute_enstrophy(conservative_soln,conservative_soln_gradient);

    }
    if (computed_quantities.size()-1 != current_data_index) {
        this->pcout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> NavierStokes<dim,nstate,real>
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
    interpretation.push_back (DCI::component_is_scalar); // Total Energy
    interpretation.push_back (DCI::component_is_scalar); // Pressure
    interpretation.push_back (DCI::component_is_scalar); // Pressure coefficient
    interpretation.push_back (DCI::component_is_scalar); // Temperature
    interpretation.push_back (DCI::component_is_scalar); // Entropy generation
    interpretation.push_back (DCI::component_is_scalar); // Mach number
    if constexpr(dim==3) {
        for (unsigned int d=0; d<3; ++d) {
            interpretation.push_back (DCI::component_is_part_of_vector); // Vorticity
        }
    }
    interpretation.push_back (DCI::component_is_scalar); // Vorticity magnitude
    interpretation.push_back (DCI::component_is_scalar); // Enstrophy

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        this->pcout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}


template <int dim, int nstate, typename real>
std::vector<std::string> NavierStokes<dim,nstate,real>
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
    names.push_back ("total_energy");
    names.push_back ("pressure");
    names.push_back ("pressure_coeffcient");
    names.push_back ("temperature");

    names.push_back ("entropy_generation");
    names.push_back ("mach_number");
    if constexpr(dim==3) {
        for (unsigned int d=0; d<3; ++d) {
            names.push_back ("vorticity");
        }
    }
    names.push_back ("vorticity_magnitude");
    names.push_back ("enstrophy");
    return names;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags NavierStokes<dim,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           | dealii::update_gradients
           ;
}

// Instantiate explicitly
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

//==============================================================================
// -> Templated member functions:
//------------------------------------------------------------------------------
// -->Required templated member functions by unit tests
//------------------------------------------------------------------------------
// -- compute_scaled_viscosity_coefficient()
template double     NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_scaled_viscosity_coefficient< double     >(const std::array<double    ,PHILIP_DIM+2> &primitive_soln) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType   >::compute_scaled_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
template RadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_scaled_viscosity_coefficient< RadType    >(const std::array<RadType   ,PHILIP_DIM+2> &primitive_soln) const;
template FadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_scaled_viscosity_coefficient< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln) const;
template RadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_scaled_viscosity_coefficient< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_scaled_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_scaled_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_scaled_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_scaled_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
//------------------------------------------------------------------------------
// -->Required templated member functions by classes derived from ModelBase or FlowSolverCaseBase
//------------------------------------------------------------------------------
// -- compute_wall_shear_stress()
template double     NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_wall_shear_stress<double    >(const std::array<double    ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &conservative_soln_gradient, const dealii::Tensor<1,PHILIP_DIM,double    > &normal_vector) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_wall_shear_stress<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient, const dealii::Tensor<1,PHILIP_DIM,FadType   > &normal_vector) const;
template RadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_wall_shear_stress<RadType   >(const std::array<RadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &conservative_soln_gradient, const dealii::Tensor<1,PHILIP_DIM,RadType   > &normal_vector) const;
template FadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_wall_shear_stress<FadFadType>(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &conservative_soln_gradient, const dealii::Tensor<1,PHILIP_DIM,FadFadType> &normal_vector) const;
template RadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_wall_shear_stress<RadFadType>(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &conservative_soln_gradient, const dealii::Tensor<1,PHILIP_DIM,RadFadType> &normal_vector) const;
// -- scale_viscosity_coefficient()
template double     NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::scale_viscosity_coefficient<double    > (const double     viscosity_coefficient) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::scale_viscosity_coefficient<FadType   > (const FadType    viscosity_coefficient) const;
template RadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::scale_viscosity_coefficient<RadType   > (const RadType    viscosity_coefficient) const;
template FadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::scale_viscosity_coefficient<FadFadType> (const FadFadType viscosity_coefficient) const;
template RadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::scale_viscosity_coefficient<RadFadType> (const RadFadType viscosity_coefficient) const;
// -- extract_velocities_gradient_from_primitive_solution_gradient()
template dealii::Tensor<2,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::extract_velocities_gradient_from_primitive_solution_gradient<double    > (const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::extract_velocities_gradient_from_primitive_solution_gradient<FadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::extract_velocities_gradient_from_primitive_solution_gradient<RadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<FadFadType> (const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<RadFadType> (const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
// -- dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux()
template std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<double    >(const dealii::Tensor<1,PHILIP_DIM,double    > &vel, const dealii::Tensor<2,PHILIP_DIM,double    > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,double    > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadType   >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,FadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadType   > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<RadType   >(const dealii::Tensor<1,PHILIP_DIM,RadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,RadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,RadType   > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadFadType>(const dealii::Tensor<1,PHILIP_DIM,FadFadType> &vel, const dealii::Tensor<2,PHILIP_DIM,FadFadType> &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadFadType> &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<RadFadType>(const dealii::Tensor<1,PHILIP_DIM,RadFadType> &vel, const dealii::Tensor<2,PHILIP_DIM,RadFadType> &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,RadFadType> &heat_flux) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadType   >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,FadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadType   > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadType   >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,FadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadType   > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadType   >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,FadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadType   > &heat_flux) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux<FadType   >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &vel, const dealii::Tensor<2,PHILIP_DIM,FadType   > &viscous_stress_tensor, const dealii::Tensor<1,PHILIP_DIM,FadType   > &heat_flux) const;
// -- compute_strain_rate_tensor()
template dealii::Tensor<2,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_strain_rate_tensor<double    > (const dealii::Tensor<2,PHILIP_DIM,double    > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_strain_rate_tensor<FadType   > (const dealii::Tensor<2,PHILIP_DIM,FadType   > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_strain_rate_tensor<RadType   > (const dealii::Tensor<2,PHILIP_DIM,RadType   > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_strain_rate_tensor<FadFadType> (const dealii::Tensor<2,PHILIP_DIM,FadFadType> &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_strain_rate_tensor<RadFadType> (const dealii::Tensor<2,PHILIP_DIM,RadFadType> &vel_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_strain_rate_tensor<FadType   > (const dealii::Tensor<2,PHILIP_DIM,FadType   > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_strain_rate_tensor<FadType   > (const dealii::Tensor<2,PHILIP_DIM,FadType   > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_strain_rate_tensor<FadType   > (const dealii::Tensor<2,PHILIP_DIM,FadType   > &vel_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_strain_rate_tensor<FadType   > (const dealii::Tensor<2,PHILIP_DIM,FadType   > &vel_gradient) const;
// -- compute_viscous_stress_tensor_from_conservative_templated()
template dealii::Tensor<2,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_viscous_stress_tensor_from_conservative_templated<double    >(const std::array<double    ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_viscous_stress_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_viscous_stress_tensor_from_conservative_templated<RadType   >(const std::array<RadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_viscous_stress_tensor_from_conservative_templated<FadFadType>(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_viscous_stress_tensor_from_conservative_templated<RadFadType>(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_viscous_stress_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_viscous_stress_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_viscous_stress_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_viscous_stress_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
// -- compute_strain_rate_tensor_from_conservative_templated()
template dealii::Tensor<2,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_strain_rate_tensor_from_conservative_templated<double    >(const std::array<double    ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_strain_rate_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_strain_rate_tensor_from_conservative_templated<RadType   >(const std::array<RadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_strain_rate_tensor_from_conservative_templated<FadFadType>(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_strain_rate_tensor_from_conservative_templated<RadFadType>(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_strain_rate_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_strain_rate_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_strain_rate_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_strain_rate_tensor_from_conservative_templated<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
// -- extract_velocities_gradient_from_primitive_solution_gradient()
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::extract_velocities_gradient_from_primitive_solution_gradient<FadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::extract_velocities_gradient_from_primitive_solution_gradient<FadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<FadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<FadType   > (const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
// -- compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor()
template dealii::Tensor<2,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<double    > (const double     scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,double    > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadType   > (const FadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadType   > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<RadType   > (const RadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,RadType   > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadFadType> (const FadFadType scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadFadType> &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<RadFadType> (const RadFadType scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,RadFadType> &strain_rate_tensor) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadType   > (const FadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadType   > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadType   > (const FadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadType   > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadType   > (const FadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadType   > &strain_rate_tensor) const;
template dealii::Tensor<2,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor<FadType   > (const FadType    scaled_viscosity_coefficient, const dealii::Tensor<2,PHILIP_DIM,FadType   > &strain_rate_tensor) const;
// -- compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient()
template dealii::Tensor<1,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<double    > (const double     scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,double    > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadType   > (const FadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadType   > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<RadType   > (const RadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,RadType   > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadFadType> (const FadFadType scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadFadType> &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<RadFadType> (const RadFadType scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,RadFadType> &temperature_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadType   > (const FadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadType   > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadType   > (const FadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadType   > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadType   > (const FadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadType   > &temperature_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient<FadType   > (const FadType    scaled_heat_conductivity, const dealii::Tensor<1,PHILIP_DIM,FadType   > &temperature_gradient) const;
// -- compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number()
template double     NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<double    > (const double     scaled_viscosity_coefficient, const double prandtl_number_input) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadType   > (const FadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
template RadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<RadType   > (const RadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
template FadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadFadType> (const FadFadType scaled_viscosity_coefficient, const double prandtl_number_input) const;
template RadFadType NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<RadFadType> (const RadFadType scaled_viscosity_coefficient, const double prandtl_number_input) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadType   > (const FadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadType   > (const FadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadType   > (const FadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
template FadType    NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number<FadType   > (const FadType    scaled_viscosity_coefficient, const double prandtl_number_input) const;
// -- compute_temperature_gradient()
template dealii::Tensor<1,PHILIP_DIM,double    > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_temperature_gradient<double    >(const std::array<double    ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadType   >::compute_temperature_gradient<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_temperature_gradient<RadType   >(const std::array<RadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_temperature_gradient<FadFadType>(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadFadType> NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_temperature_gradient<RadFadType>(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from LargeEddySimulationBase
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,double    >::compute_temperature_gradient<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadType   >::compute_temperature_gradient<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,FadFadType>::compute_temperature_gradient<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > NavierStokes<PHILIP_DIM,PHILIP_DIM+2,RadFadType>::compute_temperature_gradient<FadType   >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &primitive_soln_gradient) const;
// -- compute_viscosity_coefficient()
template double     NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_viscosity_coefficient< double     >(const std::array<double    ,PHILIP_DIM+2> &primitive_soln) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType   >::compute_viscosity_coefficient< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &primitive_soln) const;
template RadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_viscosity_coefficient< RadType    >(const std::array<RadType   ,PHILIP_DIM+2> &primitive_soln) const;
template FadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_viscosity_coefficient< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln) const;
template RadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_viscosity_coefficient< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from ModelBase
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_viscosity_coefficient< FadType >(const std::array<FadType,PHILIP_DIM+2> &primitive_soln) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_viscosity_coefficient< FadType >(const std::array<FadType,PHILIP_DIM+2> &primitive_soln) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_viscosity_coefficient< FadType >(const std::array<FadType,PHILIP_DIM+2> &primitive_soln) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_viscosity_coefficient< FadType >(const std::array<FadType,PHILIP_DIM+2> &primitive_soln) const;
// -- compute_viscosity_coefficient_from_temperature()
template double     NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_viscosity_coefficient_from_temperature< double     >(const double     temperature) const;
template FadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType   >::compute_viscosity_coefficient_from_temperature< FadType    >(const FadType    temperature) const;
template RadType    NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_viscosity_coefficient_from_temperature< RadType    >(const RadType    temperature) const;
template FadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_viscosity_coefficient_from_temperature< FadFadType >(const FadFadType temperature) const;
template RadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_viscosity_coefficient_from_temperature< RadFadType >(const RadFadType temperature) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from ModelBase
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_viscosity_coefficient_from_temperature< FadType >(const FadType temperature) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_viscosity_coefficient_from_temperature< FadType >(const FadType temperature) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_viscosity_coefficient_from_temperature< FadType >(const FadType temperature) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_viscosity_coefficient_from_temperature< FadType >(const FadType temperature) const;
// -- compute_vorticity()
template dealii::Tensor<1,3,double    > NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_vorticity< double     >(const std::array<double    ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,FadType   > NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType   >::compute_vorticity< FadType    >(const std::array<FadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,RadType   > NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_vorticity< RadType    >(const std::array<RadType   ,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,FadFadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_vorticity< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,RadFadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_vorticity< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in classes derived from ModelBase
template dealii::Tensor<1,3,FadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double    >::compute_vorticity< FadType >(const std::array<FadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,FadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType   >::compute_vorticity< FadType >(const std::array<FadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,FadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_vorticity< FadType >(const std::array<FadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;
template dealii::Tensor<1,3,FadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_vorticity< FadType >(const std::array<FadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &conservative_soln_gradient) const;


} // Physics namespace
} // PHiLiP namespace
