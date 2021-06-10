#include <cmath>
#include <vector>

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
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : Euler<dim,nstate,real>(ref_length, 
                             gamma_gas, 
                             mach_inf, 
                             angle_of_attack, 
                             side_slip_angle, 
                             input_diffusion_tensor, 
                             manufactured_solution_function)
    , viscosity_coefficient_inf(1.0) // Nondimensional - Free stream values
    , prandtl_number(prandtl_number)
    , reynolds_number_inf(reynolds_number_inf)
{
    // Nothing to do here so far
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes<dim,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // conservative_soln_gradient is solution_gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient;

    // get primitive solution
    const std::array<real,nstate> primitive_soln = this->convert_conservative_to_primitive(conservative_soln); // from Euler
    // extract from primitive solution
    const real density = primitive_soln[0];
    const dealii::Tensor<1,dim,real> vel = this->extract_velocities_from_primitive(primitive_soln); // from Euler
    const real vel2 = this->compute_velocity_squared(vel); // from Euler

    // density gradient
    for (int d=0; d<dim; d++) {
        primitive_soln_gradient[0][d] = conservative_soln_gradient[0][d];
    }
    // velocities gradient
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            primitive_soln_gradient[1+d1][d2] = (conservative_soln_gradient[1+d1][d2] - vel[d1]*conservative_soln_gradient[0][d2])/density;
        }        
    }
    // pressure gradient
    for (int d1=0; d1<dim; d1++) {
        primitive_soln_gradient[nstate-1][d1] = conservative_soln_gradient[nstate-1][d1] - 0.5*vel2*conservative_soln_gradient[0][d1];
        for (int d2=0; d2<dim; d2++) {
            primitive_soln_gradient[nstate-1][d1] -= conservative_soln[1+d2]*primitive_soln_gradient[1+d2][d1];
        }
        primitive_soln_gradient[nstate-1][d1] *= this->gamm1;
    }
    return primitive_soln_gradient;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes<dim,nstate,real>
::compute_temperature_gradient (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    const real density = primitive_soln[0];
    const real temperature = this->compute_temperature(primitive_soln); // from Euler

    dealii::Tensor<1,dim,real> temperature_gradient;
    for (int d=0; d<dim; d++) {
        temperature_gradient[d] = (this->gam*this->mach_inf_sqr*primitive_soln_gradient[nstate-1][d] - temperature*primitive_soln_gradient[0][d])/density;
    }
    return temperature_gradient;
}

template <int dim, int nstate, typename real>
inline real NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const real temperature = this->compute_temperature(primitive_soln); // from Euler

    const real viscosity_coefficient = ((1.0 + temperature_ratio)/(temperature + temperature_ratio))*pow(temperature,1.5);
    
    return viscosity_coefficient;
}

template <int dim, int nstate, typename real>
inline real NavierStokes<dim,nstate,real>
::compute_scaled_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const real viscosity_coefficient = compute_viscosity_coefficient(primitive_soln);
    const real scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;

    return scaled_viscosity_coefficient;
}

template <int dim, int nstate, typename real>
inline real NavierStokes<dim,nstate,real>
::compute_scaled_heat_conductivity (const std::array<real,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const real scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    const real scaled_heat_conductivity = scaled_viscosity_coefficient/(this->gamm1*this->mach_inf_sqr*prandtl_number);
    
    return scaled_heat_conductivity;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes<dim,nstate,real>
::compute_heat_flux (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const real scaled_heat_conductivity = compute_scaled_heat_conductivity(primitive_soln);
    const dealii::Tensor<1,dim,real> temperature_gradient = compute_temperature_gradient(primitive_soln, primitive_soln_gradient);

    dealii::Tensor<1,dim,real> heat_flux;
    for (int d=0; d<dim; d++) {
        heat_flux[d] = -scaled_heat_conductivity*temperature_gradient[d];
    }
    return heat_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> NavierStokes<dim,nstate,real>
::extract_velocities_gradient_from_primitive_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,dim> velocities_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            velocities_gradient[d1][d2] = primitive_soln_gradient[1+d1][d2];
        }
    }
    return velocities_gradient;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,dim> NavierStokes<dim,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    const std::array<dealii::Tensor<1,dim,real>,dim> vel_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const real scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);
    
    // Divergence of velocity
    real vel_divergence = 0.0;
    for (int d=0; d<dim; d++) {
        vel_divergence += vel_gradient[d][d];
    }

    // Viscous stress tensor, \tau_{i,j}
    std::array<dealii::Tensor<1,dim,real>,dim> viscous_stress_tensor;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            // rate of strain (deformation) tensor:
            viscous_stress_tensor[d1][d2] = scaled_viscosity_coefficient*(vel_gradient[d1][d2] + vel_gradient[d2][d1]);
        }
        viscous_stress_tensor[d1][d1] += (-2.0/3.0)*scaled_viscosity_coefficient*vel_divergence;
    }
    return viscous_stress_tensor;
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

    // Step 1: Primitive solution
    const std::array<real,nstate> primitive_soln = this->convert_conservative_to_primitive(conservative_soln); // from Euler
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->convert_conservative_gradient_to_primitive_gradient(conservative_soln, solution_gradient);
    
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const std::array<dealii::Tensor<1,dim,real>,dim> viscous_stress_tensor = compute_viscous_stress_tensor(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real> vel = this->extract_velocities_from_primitive(primitive_soln); // from Euler
    const dealii::Tensor<1,dim,real> heat_flux = compute_heat_flux(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux;
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
dealii::Tensor<1,dim,real> NavierStokes<dim,nstate,real>
::compute_scaled_viscosity_gradient (
    const std::array<real,nstate> &primitive_soln,
    const dealii::Tensor<1,dim,real> temperature_gradient) const
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

// Instantiate explicitly
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace