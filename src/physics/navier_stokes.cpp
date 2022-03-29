#include <cmath>
#include <vector>
#include <complex> // for the jacobian

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
    static_assert(nstate==dim+2, "Physics::NavierStokes() should be created with nstate=dim+2");
    // Nothing to do here so far
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> NavierStokes<dim,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const
{
    // conservative_soln_gradient is solution_gradient
    std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient;

    // get primitive solution
    const std::array<real2,nstate> primitive_soln = this->template convert_conservative_to_primitive<real2>(conservative_soln); // from Euler
    // extract from primitive solution
    const real2 density = primitive_soln[0];
    const dealii::Tensor<1,dim,real2> vel = this->template extract_velocities_from_primitive<real2>(primitive_soln); // from Euler
    const real2 vel2 = this->template compute_velocity_squared<real2>(vel); // from Euler

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
inline real2 NavierStokes<dim,nstate,real>
::compute_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const real2 temperature = this->template compute_temperature<real2>(primitive_soln); // from Euler

    const real2 viscosity_coefficient = ((1.0 + temperature_ratio)/(temperature + temperature_ratio))*pow(temperature,1.5);
    
    return viscosity_coefficient;
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
    const real2 scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;
    // print the value for Re
    // std::cout << "\n Reynolds number inside compute_scaled_viscosity_coefficient(): " << reynolds_number_inf << "\n" << std::endl;

    return scaled_viscosity_coefficient;
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

    const real2 scaled_heat_conductivity = scaled_viscosity_coefficient/(this->gamm1*this->mach_inf_sqr*prandtl_number);
    
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

    dealii::Tensor<1,dim,real2> heat_flux;
    for (int d=0; d<dim; d++) {
        heat_flux[d] = -scaled_heat_conductivity*temperature_gradient[d];
    }
    return heat_flux;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> NavierStokes<dim,nstate,real>
::extract_velocities_gradient_from_primitive_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    std::array<dealii::Tensor<1,dim,real2>,dim> velocities_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            velocities_gradient[d1][d2] = primitive_soln_gradient[1+d1][d2];
        }
    }
    return velocities_gradient;
}

template <int dim, int nstate, typename real>
template<typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim> NavierStokes<dim,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    const std::array<dealii::Tensor<1,dim,real2>,dim> vel_gradient = extract_velocities_gradient_from_primitive_solution_gradient<real2>(primitive_soln_gradient);
    const real2 scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient<real2>(primitive_soln);
    
    // Divergence of velocity
    real2 vel_divergence; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){ 
        vel_divergence = 0.0;
    }
    for (int d=0; d<dim; d++) {
        vel_divergence += vel_gradient[d][d];
    }

    // Viscous stress tensor, \tau_{i,j}
    std::array<dealii::Tensor<1,dim,real2>,dim> viscous_stress_tensor;
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
    std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux = dissipative_flux_templated<real>(conservative_soln, solution_gradient);
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
    std::array<real,nstate> &conservative_soln,
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
    std::array<real,nstate> &conservative_soln,
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
    /* START: Repeated code */
    // Get MS values
    std::array<real,nstate> manufactured_solution;
    for (int s=0; s<nstate; s++) {
        manufactured_solution[s] = this->manufactured_solution_function->value(pos,s);
        if (s==0) {
            assert(manufactured_solution[s] > 0);
        }
    }
    // Get MS gradient
    std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
    this->manufactured_solution_function->vector_gradient(pos,manufactured_solution_gradient_dealii);
    std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient;
    for (int d=0;d<dim;d++) {
        for (int s=0; s<nstate; s++) {
            manufactured_solution_gradient[s][d] = manufactured_solution_gradient_dealii[s][d]; // CHANGED THE INDEXING HERE
        }
    }
    /* END of repeated code */
    // Get MS hessian
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

        //dissipative_flux_divergence += jacobian*manufactured_solution_gradient[d]; <-- needs second term! (jac wrt gradient)
        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d]; // CHANGED THE INDEXING HERE
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
    const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive<real>(conservative_soln); // from Euler
    
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
    const std::array<real2,nstate> primitive_soln = this->template convert_conservative_to_primitive<real2>(conservative_soln); // from Euler
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient<real2>(conservative_soln, solution_gradient);
    
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const std::array<dealii::Tensor<1,dim,real2>,dim> viscous_stress_tensor = compute_viscous_stress_tensor<real2>(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real2> vel = this->template extract_velocities_from_primitive<real2>(primitive_soln); // from Euler
    const dealii::Tensor<1,dim,real2> heat_flux = compute_heat_flux<real2>(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
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
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1000) {
        // Manufactured solution boundary condition 
        // -- TO DO: could make this a function later on, similar to how its done in euler
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
    // else if (boundary_type == 1005) {
    //     // Simple farfield boundary condition
    //     this->boundary_farfield(soln_bc);
    // }
    else {
        std::cout << "Invalid boundary_type: " << boundary_type << std::endl;
        std::abort();
    }
}

// Instantiate explicitly
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

// Templated member functions:
template double NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double>::compute_scaled_viscosity_coefficient< double >(const std::array<double,PHILIP_DIM+2> &primitive_soln) const;
template FadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType>::compute_scaled_viscosity_coefficient< FadType >(const std::array<FadType,PHILIP_DIM+2> &primitive_soln) const;
template RadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType>::compute_scaled_viscosity_coefficient< RadType >(const std::array<RadType,PHILIP_DIM+2> &primitive_soln) const;
template FadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_scaled_viscosity_coefficient< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln) const;
template RadFadType NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_scaled_viscosity_coefficient< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln) const;

template std::array<dealii::Tensor<1,PHILIP_DIM,double>,PHILIP_DIM+2>  NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double>::convert_conservative_gradient_to_primitive_gradient< double >(const std::array<double,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1, PHILIP_DIM, double>, PHILIP_DIM+2> &conservative_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2>  NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType>::convert_conservative_gradient_to_primitive_gradient< FadType >(const std::array<FadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1, PHILIP_DIM, FadType>, PHILIP_DIM+2> &conservative_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadType>,PHILIP_DIM+2> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType>::convert_conservative_gradient_to_primitive_gradient< RadType >(const std::array<RadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1, PHILIP_DIM, RadType>, PHILIP_DIM+2> &conservative_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::convert_conservative_gradient_to_primitive_gradient< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1, PHILIP_DIM, FadFadType>, PHILIP_DIM+2> &conservative_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::convert_conservative_gradient_to_primitive_gradient< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln, const std::array<dealii::Tensor<1, PHILIP_DIM, RadFadType>, PHILIP_DIM+2> &conservative_soln_gradient) const;

template dealii::Tensor<1,PHILIP_DIM,double> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double>::compute_heat_flux<double>(const std::array<double,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType>::compute_heat_flux<FadType>(const std::array<FadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType>::compute_heat_flux<RadType>(const std::array<RadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,FadFadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::compute_heat_flux<FadFadType>(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template dealii::Tensor<1,PHILIP_DIM,RadFadType> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::compute_heat_flux<RadFadType>(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;

template std::array<dealii::Tensor<1,PHILIP_DIM,double>,PHILIP_DIM> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, double>::extract_velocities_gradient_from_primitive_solution_gradient<double>(const std::array<dealii::Tensor<1,PHILIP_DIM,double>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadType>::extract_velocities_gradient_from_primitive_solution_gradient<FadType>(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadType>,PHILIP_DIM> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadType>::extract_velocities_gradient_from_primitive_solution_gradient<RadType>(const std::array<dealii::Tensor<1,PHILIP_DIM,RadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, FadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<FadFadType>(const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM> NavierStokes < PHILIP_DIM, PHILIP_DIM+2, RadFadType>::extract_velocities_gradient_from_primitive_solution_gradient<RadFadType>(const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> &primitive_soln_gradient) const;
} // Physics namespace
} // PHiLiP namespace
