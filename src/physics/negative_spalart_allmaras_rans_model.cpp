#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "model.h"
#include "reynolds_averaged_navier_stokes.h"
#include "negative_spalart_allmaras_rans_model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Negative Spalart-Allmaras model
//================================================================
template <int dim, int nstate, typename real>
ReynoldsAveragedNavierStokes_SAneg<dim, nstate, real>::ReynoldsAveragedNavierStokes_SAneg(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              temperature_inf,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : ReynoldsAveragedNavierStokesBase<dim,nstate,real>(ref_length,
                                                        gamma_gas,
                                                        mach_inf,
                                                        angle_of_attack,
                                                        side_slip_angle,
                                                        prandtl_number,
                                                        reynolds_number_inf,
                                                        turbulent_prandtl_number,
                                                        temperature_inf,
                                                        isothermal_wall_temperature,
                                                        thermal_boundary_condition_type,
                                                        manufactured_solution_function,
                                                        two_point_num_flux_type)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_fad (
    const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_eddy_viscosity_templated<FadType>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<real2,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    real2 eddy_viscosity;
    if (primitive_soln_turbulence_model[0]>=0.0)
    {
        // Compute needed coefficients
        const real2 laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
        const real2 laminar_kinematic_viscosity = laminar_dynamic_viscosity/primitive_soln_rans[0];
        const real2 Chi = this->compute_coefficient_Chi(primitive_soln_turbulence_model[0],laminar_kinematic_viscosity);
        const real2 f_v1 = this->compute_coefficient_f_v1(Chi);
        eddy_viscosity = primitive_soln_rans[0]*primitive_soln_turbulence_model[0]*f_v1;
    } else {
        eddy_viscosity = 0.0;
    }

    return eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::scale_coefficient (
    const real2 coefficient) const
{
    real2 scaled_coefficient;
    if constexpr(std::is_same<real2,real>::value){
        scaled_coefficient = coefficient/this->navier_stokes_physics->reynolds_number_inf;
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType reynolds_number_inf_fad = this->navier_stokes_physics->reynolds_number_inf;
        scaled_coefficient = coefficient/reynolds_number_inf_fad;
    }
    else{
        std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> scale_coefficient(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return scaled_coefficient;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model (
    const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{   
    return compute_effective_viscosity_turbulence_model_templated<real>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<FadType,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model_fad (
    const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{   
    return compute_effective_viscosity_turbulence_model_templated<FadType>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model_templated (
    const std::array<real2,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<real2,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{   
    const real2 laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
    const real2 laminar_kinematic_viscosity = laminar_dynamic_viscosity/primitive_soln_rans[0];

    const real2 coefficient_f_n = this->compute_coefficient_f_n(primitive_soln_turbulence_model[0],laminar_kinematic_viscosity);

    std::array<real2,nstate_turbulence_model> effective_viscosity_turbulence_model;

    for(int i=0; i<nstate_turbulence_model; ++i){
        if constexpr(std::is_same<real2,real>::value){
            effective_viscosity_turbulence_model[i] = (laminar_dynamic_viscosity+coefficient_f_n*primitive_soln_rans[0]*primitive_soln_turbulence_model[0])/sigma;
        }
        else if constexpr(std::is_same<real2,FadType>::value){
            effective_viscosity_turbulence_model[i] = (laminar_dynamic_viscosity+coefficient_f_n*primitive_soln_rans[0]*primitive_soln_turbulence_model[0])/sigma_fad;
        }
        else{
            std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> compute_effective_viscosity_turbulence_model_templated(): real2 != real or FadType" << std::endl;
            std::abort();
        }
        effective_viscosity_turbulence_model[i] = scale_coefficient(effective_viscosity_turbulence_model[i]);
    }
    
    return effective_viscosity_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_Chi (
    const real2 nu_tilde,
    const real2 laminar_kinematic_viscosity) const
{
    // Compute coefficient Chi
    const real2 Chi = nu_tilde/laminar_kinematic_viscosity;

    return Chi;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_v1 (
    const real2 coefficient_Chi) const
{
    // Compute coefficient f_v1
    const real2 coefficient_Chi_power_3 = pow(coefficient_Chi,3.0);
    real2 coefficient_f_v1;

    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_f_v1 = coefficient_Chi_power_3/(coefficient_Chi_power_3+c_v1*c_v1*c_v1);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        coefficient_f_v1 = coefficient_Chi_power_3/(coefficient_Chi_power_3+c_v1_fad*c_v1_fad*c_v1_fad);
    }
    else{
        std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> compute_coefficient_f_v1(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_v1;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_v2 (
    const real coefficient_Chi) const
{
    // Compute coefficient f_v2
    real coefficient_f_v2;
    const real coefficient_f_v1 = compute_coefficient_f_v1(coefficient_Chi);

    coefficient_f_v2 = 1.0-coefficient_Chi/(1.0+coefficient_Chi*coefficient_f_v1);

    return coefficient_f_v2;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_n (
    const real2 nu_tilde,
    const real2 laminar_kinematic_viscosity) const
{
    // Compute coefficient f_n
    real2 coefficient_f_n;
    const real2 coefficient_Chi_power_3 = pow(compute_coefficient_Chi(nu_tilde,laminar_kinematic_viscosity),3.0);

    if constexpr(std::is_same<real2,real>::value){ 
        if (nu_tilde>=0.0)
            coefficient_f_n = 1.0;
        else
            coefficient_f_n = (c_n1+coefficient_Chi_power_3)/(c_n1-coefficient_Chi_power_3);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType const_one_fad = 1.0; 
        if (nu_tilde>=0.0)
            coefficient_f_n = const_one_fad;
        else
            coefficient_f_n = (c_n1_fad+coefficient_Chi_power_3)/(c_n1_fad-coefficient_Chi_power_3);
    }
    else{
        std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> compute_coefficient_f_n(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_n;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_t2 (
    const real coefficient_Chi) const
{
    // Compute coefficient f_t2
    real coefficient_f_t2;

    coefficient_f_t2 = c_t3*exp(-c_t4*coefficient_Chi*coefficient_Chi);

    return coefficient_f_t2;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_w (
    const real coefficient_g) const
{
    // Compute coefficient f_w
    real coefficient_f_w;
    const real cw3_power_6 = pow(c_w3,6.0);
    coefficient_f_w = coefficient_g*pow((1.0+cw3_power_6)/(pow(coefficient_g,6.0)+cw3_power_6),1.0/6.0);

    return coefficient_f_w;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_r (
    const real nu_tilde,
    const real d_wall,
    const real s_tilde) const
{
    // Compute coefficient r
    real coefficient_r;
    real dimensional_r;

    if (s_tilde<=0.0){
        coefficient_r = r_lim;
    } else{
        dimensional_r = nu_tilde/(s_tilde*kappa*kappa*d_wall*d_wall); 
        dimensional_r = scale_coefficient(dimensional_r);
        coefficient_r = dimensional_r <= r_lim ? dimensional_r : r_lim; 
    }

    return coefficient_r;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_g (
    const real coefficient_r) const
{
    // Compute coefficient g
    real coefficient_g;

    coefficient_g = coefficient_r+c_w2*(pow(coefficient_r,6.0)-coefficient_r);

    return coefficient_g;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s (
    const std::array<real,nstate_navier_stokes> &conservative_soln_rans,
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &conservative_soln_gradient_rans) const
{
    // Compute s (non-dimensional)
    real s;

    // Get vorticity
    const dealii::Tensor<1,3,real> vorticity 
        = this->navier_stokes_physics->compute_vorticity(conservative_soln_rans,conservative_soln_gradient_rans);

    s = sqrt(this->get_vector_magnitude_sqr(vorticity));

    return s;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s_bar (
    const real coefficient_Chi,
    const real nu_tilde,
    const real d_wall) const
{
    // Compute s_bar (non-dimensional)
    real s_bar;
    const real f_v2 = compute_coefficient_f_v2(coefficient_Chi);

    s_bar = nu_tilde*f_v2/(kappa*kappa*d_wall*d_wall);

    return s_bar;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s_tilde (
    const real coefficient_Chi,
    const real nu_tilde,
    const real d_wall,
    const real s) const
{
    // Compute s_tilde
    real s_tilde;
    const real s_bar = compute_s_bar(coefficient_Chi,nu_tilde,d_wall);
    const real scaled_s_bar = scale_coefficient(s_bar);


    const real dimensional_s = s*this->navier_stokes_physics->mach_inf/this->navier_stokes_physics->ref_length;
    const real dimensional_s_bar = s_bar*this->navier_stokes_physics->mach_inf/(this->navier_stokes_physics->reynolds_number_inf*this->navier_stokes_physics->ref_length);
    if(dimensional_s_bar>=-c_v2*dimensional_s) 
        s_tilde = s+scaled_s_bar;
    else
        s_tilde = s+s*(c_v2*c_v2*s+c_v3*scaled_s_bar)/((c_v3-2.0*c_v2)*s-scaled_s_bar);

    return s_tilde;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_source (
    const real coefficient_f_t2,
    const real density,
    const real nu_tilde,
    const real s,
    const real s_tilde) const
{
    std::array<real,nstate> production_source;

    for (int i=0;i<nstate_navier_stokes;++i){
        production_source[i] = 0.0;
    }

    if(nu_tilde>=0.0)
        production_source[nstate_navier_stokes] = c_b1*(1.0-coefficient_f_t2)*s_tilde*nu_tilde;
    else
        production_source[nstate_navier_stokes] = c_b1*(1.0-c_t3)*s*nu_tilde;

    production_source[nstate_navier_stokes] *= density;

    return production_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_dissipation_source (
    const real coefficient_f_t2,
    const real density,
    const real nu_tilde,
    const real d_wall,
    const real s_tilde) const
{
    const real coefficient_r = this->compute_coefficient_r(nu_tilde,d_wall,s_tilde);
    const real coefficient_g = this->compute_coefficient_g(coefficient_r);
    const real coefficient_f_w = this->compute_coefficient_f_w(coefficient_g);
    std::array<real,nstate> dissipation_source;

    for (int i=0;i<nstate_navier_stokes;++i){
        dissipation_source[i] = 0.0;
    }

    if(nu_tilde>=0.0)
        dissipation_source[nstate_navier_stokes] = (c_w1*coefficient_f_w-c_b1*coefficient_f_t2/(kappa*kappa))*nu_tilde*nu_tilde/(d_wall*d_wall);
    else
        dissipation_source[nstate_navier_stokes] = -c_w1*nu_tilde*nu_tilde/(d_wall*d_wall);

    dissipation_source[nstate_navier_stokes] *= density;
    dissipation_source[nstate_navier_stokes] = scale_coefficient(dissipation_source[nstate_navier_stokes]);

    return dissipation_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_cross_source (
    const real density,
    const real nu_tilde,
    const real laminar_kinematic_viscosity,
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<dealii::Tensor<1,dim,real>,nstate_turbulence_model> &primitive_solution_gradient_turbulence_model) const
{
    real cross_nu_tilde_nu_tilde = 0.0;
    real cross_rho_nu_tilde = 0.0;
    std::array<real,nstate> cross_source;
    const real coefficient_f_n = this->compute_coefficient_f_n(nu_tilde,laminar_kinematic_viscosity);

    for (int i=0;i<nstate_navier_stokes;++i){
        cross_source[i] = 0.0;
    }
    for (int i=0;i<dim;++i){
        cross_nu_tilde_nu_tilde += primitive_solution_gradient_turbulence_model[0][i]*primitive_solution_gradient_turbulence_model[0][i];
        cross_rho_nu_tilde += primitive_soln_gradient_rans[0][i]*primitive_solution_gradient_turbulence_model[0][i];
    }

    cross_source[nstate_navier_stokes] = (c_b2*density*cross_nu_tilde_nu_tilde-(laminar_kinematic_viscosity+nu_tilde*coefficient_f_n)*cross_rho_nu_tilde)/sigma;
    cross_source[nstate_navier_stokes] = scale_coefficient(cross_source[nstate_navier_stokes]);

    return cross_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux (
    const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_heat_flux_templated<real>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_fad (
    const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_heat_flux_templated<FadType>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_templated (
    const std::array<real2,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real2>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<real2,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{   
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> compute_Reynolds_heat_flux_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    // Scaled non-dimensional eddy viscosity;
    const real2 scaled_eddy_viscosity = scale_coefficient(eddy_viscosity);

    // Compute scaled heat conductivity
    const real2 scaled_heat_conductivity = this->navier_stokes_physics->compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_eddy_viscosity,this->turbulent_prandtl_number);

    // Get temperature gradient
    const dealii::Tensor<1,dim,real2> temperature_gradient = this->navier_stokes_physics->compute_temperature_gradient(primitive_soln_rans, primitive_soln_gradient_rans);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<1,dim,real2> heat_flux_Reynolds = this->navier_stokes_physics->compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);

    return heat_flux_Reynolds;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor (
    const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_stress_tensor_templated<real>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_fad (
    const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_stress_tensor_templated<FadType>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_templated (
    const std::array<real2,nstate_navier_stokes> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real2>,nstate_navier_stokes> &primitive_soln_gradient_rans,
    const std::array<real2,nstate_turbulence_model> &primitive_soln_turbulence_model) const
{
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/negative_spalart_allmaras_rans_model.cpp --> compute_Reynolds_stress_tensor_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }
    
    // Scaled non-dimensional eddy viscosity; 
    const real2 scaled_eddy_viscosity = scale_coefficient(eddy_viscosity);

    // Get velocity gradients
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient_rans);
    
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<2,dim,real2> Reynolds_stress_tensor;
    Reynolds_stress_tensor = this->navier_stokes_physics->compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor(scaled_eddy_viscosity,strain_rate_tensor);

    return Reynolds_stress_tensor;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_dissipation_cross_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_gradient,
    const real post_processed_scalar) const
{

    const std::array<real,nstate_navier_stokes> conservative_soln_rans = this->extract_rans_conservative_solution(conservative_soln);
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> conservative_soln_gradient_rans = this->extract_rans_solution_gradient(soln_gradient);
    const std::array<real,nstate_navier_stokes> primitive_soln_rans = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln_rans); // from Euler
    const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> primitive_soln_gradient_rans = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient(conservative_soln_rans, conservative_soln_gradient_rans);
    const std::array<dealii::Tensor<1,dim,real>,nstate_turbulence_model> primitive_soln_gradient_turbulence_model = this->convert_conservative_gradient_to_primitive_gradient_turbulence_model(conservative_soln, soln_gradient);


    const real density = conservative_soln_rans[0];
    const real nu_tilde = conservative_soln[nstate_navier_stokes]/conservative_soln_rans[0];
    const real laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
    const real laminar_kinematic_viscosity = laminar_dynamic_viscosity/density;

    const real coefficient_Chi = compute_coefficient_Chi(nu_tilde,laminar_kinematic_viscosity);
    const real coefficient_f_t2 = compute_coefficient_f_t2(coefficient_Chi); 

    const real d_wall = post_processed_scalar;

    const real s = compute_s(conservative_soln_rans, conservative_soln_gradient_rans);
    const real s_tilde = compute_s_tilde(coefficient_Chi, nu_tilde, d_wall, s);

    const std::array<real,nstate> production = compute_production_source(coefficient_f_t2, density, nu_tilde, s, s_tilde);
    const std::array<real,nstate> dissipation = compute_dissipation_source(coefficient_f_t2, density, nu_tilde, d_wall, s_tilde);
    const std::array<real,nstate> cross = compute_cross_source(density, nu_tilde, laminar_kinematic_viscosity, primitive_soln_gradient_rans, primitive_soln_gradient_turbulence_model);

    std::array<real,nstate> physical_source_term;
    for (int i=0;i<nstate;++i){
        physical_source_term[i] = production[i]-dissipation[i]+cross[i];
    }

    return physical_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_wall (
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
    // Wall boundary condition for nu_tilde (working variable of negative SA model)
    // nu_tilde = 0
    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_outflow (
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = soln_int[istate];
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_inflow (
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
    const real density_bc = this->navier_stokes_physics->density_inf;
    const real dynamic_viscosity_coefficient_bc = this->navier_stokes_physics->viscosity_coefficient_inf;
    const real kinematic_viscosity_coefficient_bc = dynamic_viscosity_coefficient_bc/density_bc;

    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = density_bc*3.0*kinematic_viscosity_coefficient_bc;
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_farfield (
   std::array<real,nstate> &soln_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
    }
    const real density_bc = this->navier_stokes_physics->density_inf;
    const real dynamic_viscosity_coefficient_bc = this->navier_stokes_physics->viscosity_coefficient_inf;
    const real kinematic_viscosity_coefficient_bc = dynamic_viscosity_coefficient_bc/density_bc;
   
    // Farfield boundary condition for nu_tilde (working variable of negative SA model)
    // nu_tilde = 3.0*kinematic_viscosity_inf to 5.0*kinematic_viscosity_inf
    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = density_bc*3.0*kinematic_viscosity_coefficient_bc;
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_slip_wall (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = soln_int[istate];
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::boundary_riemann (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &soln_int,
   std::array<real,nstate> &soln_bc) const
{
    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
        soln_bc[istate] = 0.0;
    }
    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
        soln_bc[istate] = soln_int[istate];
    }

//    (void) soln_int;
//    for (int istate=0; istate<nstate_navier_stokes; ++istate) {
//        soln_bc[istate] = 0.0;
//    }
//    const real density_bc = this->navier_stokes_physics->density_inf;
//    const real dynamic_viscosity_coefficient_bc = this->navier_stokes_physics->viscosity_coefficient_inf;
//    const real kinematic_viscosity_coefficient_bc = dynamic_viscosity_coefficient_bc/density_bc;
//   
//    // Farfield boundary condition for nu_tilde (working variable of negative SA model)
//    // nu_tilde = 3.0*kinematic_viscosity_inf to 5.0*kinematic_viscosity_inf
//    for (int istate=nstate_navier_stokes; istate<nstate; ++istate) {
//        soln_bc[istate] = density_bc*3.0*kinematic_viscosity_coefficient_bc;
//    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Vector<double> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    std::vector<std::string> names = post_get_names ();
    dealii::Vector<double> computed_quantities = ModelBase<dim,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {

        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }

        const std::array<double,nstate_turbulence_model> primitive_soln_turbulence_model = this->convert_conservative_to_primitive_turbulence_model(conservative_soln); 

        computed_quantities(++current_data_index) = primitive_soln_turbulence_model[0];
    }
    if (computed_quantities.size()-1 != current_data_index) {
        std::cout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation = ModelBase<dim,nstate,real>::post_get_data_component_interpretation (); // state variables
    interpretation.push_back (DCI::component_is_scalar); 

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        std::cout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::vector<std::string> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names = ModelBase<dim,nstate,real>::post_get_names ();
    names.push_back ("nu_tilde");

    return names;    
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- ReynoldsAveragedNavierStokes_SAneg
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, double >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, FadType  >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, RadType  >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, RadFadType >;

} // Physics namespace
} // PHiLiP namespace