#ifndef __NEGATIVE_SPALART_ALLMARAS__
#define __NEGATIVE_SPALART_ALLMARAS__

#include "euler.h"
#include "navier_stokes.h"
#include "model.h"
#include "reynolds_averaged_navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Negative Spalart-Allmaras model. Derived from Reynolds Averaged Navier Stokes.
template <int dim, int nstate, typename real>
class ReynoldsAveragedNavierStokes_SAneg : public ReynoldsAveragedNavierStokesBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    /** Constructor for the Reynolds-averaged Navier-Stokes model: negative SA
     *  Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     */
    ReynoldsAveragedNavierStokes_SAneg(
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

    /// Destructor
    ~ReynoldsAveragedNavierStokes_SAneg() {};

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*, for the negative SA model
    dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*, for the negative SA model
    dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType), for the negative SA model
    dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType), for the negative SA model
    dealii::Tensor<1,dim,FadType> compute_Reynolds_heat_flux_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized eddy viscosity for the negative SA model
    real compute_eddy_viscosity(
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized eddy viscosity for the negative SA model (Automatic Differentiation Type: FadType)
    FadType compute_eddy_viscosity_fad(
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized effective (total) viscosities for the negative SA model
    std::array<real,nstate-(dim+2)> compute_effective_viscosity_turbulence_model (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized effective (total) viscosities for the negative SA model (Automatic Differentiation Type: FadType)
    std::array<FadType,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Physical source term (production, dissipation source terms and source term with cross derivatives) for the negative SA model
    std::array<real,nstate> compute_production_dissipation_cross_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// For post processing purposes, returns conservative and primitive solution variables for the negative SA model
    dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
        const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
        const dealii::Tensor<1,dim>               &/*normals*/,
        const dealii::Point<dim>                  &/*evaluation_points*/) const override;

    /// For post processing purposes, sets the interpretation of each computed quantity as either scalar or vector for the negative SA model
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const override;

    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities for the negative SA model
    std::vector<std::string> post_get_names () const override;

protected:
    /// Templated nondimensionalized Reynolds stress tensor, (tau^reynolds)* for the negative SA model
    template<typename real2> dealii::Tensor<2,dim,real2> compute_Reynolds_stress_tensor_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated nondimensionalized Reynolds heat flux, (q^reynolds)* for the negative SA model
    template<typename real2> dealii::Tensor<1,dim,real2> compute_Reynolds_heat_flux_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated nondimensionalized variables scaled by reynolds_number_inf for the negative SA model
    template<typename real2> real2 scale_coefficient(
        const real2 coefficient) const;

private:
    /// Templated nondimensionalized eddy viscosity for the negative SA model.
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(1)
     */ 
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated nondimensionalized effective (total) viscosities for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(14)
     */ 
    template<typename real2> std::array<real2,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated coefficient Chi for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(1)
     */ 
    template<typename real2> real2 compute_coefficient_Chi (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Templated coefficient f_v1 for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(1)
     */ 
    template<typename real2> real2 compute_coefficient_f_v1 (
        const real2 &coefficient_Chi) const;

    /// Coefficient f_v2 for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(4)
     */ 
    real compute_coefficient_f_v2 (
        const real &coefficient_Chi) const;

    /// Templated coefficient f_n for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(21)
     */ 
    template<typename real2> real2 compute_coefficient_f_n (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Coefficient f_t2 for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(6)
     */ 
    real compute_coefficient_f_t2 (
        const real &coefficient_Chi) const;

    /// Coefficient f_t2 for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(5)
     */ 
    real compute_coefficient_f_w (
        const real &coefficient_g) const;

    /// Coefficient r for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(5)
     */ 
    real compute_coefficient_r (
        const real &nu_tilde,
        const real &d_wall,
        const real &s_tilde) const;

    /// Coefficient g for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(5)
     */ 
    real compute_coefficient_g (
        const real &coefficient_r) const;

    /// Vorticity magnitude for the negative SA model, sqrt(vorticity_x^2+vorticity_y^2+vorticity_z^2)
    real compute_s (
        const std::array<real,dim+2> &conservative_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &conservative_soln_gradient_rans) const;

    /// Correction of vorticity magnitude for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(11)
     */ 
    real compute_s_bar (
        const real &coefficient_Chi,
        const real &nu_tilde,
        const real &d_wall) const;

    /// Modified vorticity magnitude for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(12)
     */ 
    real compute_s_tilde (
        const real &coefficient_Chi,
        const real &nu_tilde,
        const real &d_wall,
        const real &s) const;

    /// Production source term for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(3) for positive nu_tilde
     *  eq.(22) for negative nu_tilde
     */ 
    std::array<real,nstate> compute_production_source (
        const real &coefficient_f_t2,
        const real &density,
        const real &nu_tilde,
        const real &s,
        const real &s_tilde) const;

    /// Dissipation source term for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(3) for positive nu_tilde
     *  eq.(22) for negative nu_tilde
     */ 
    std::array<real,nstate> compute_dissipation_source (
        const real &coefficient_f_t2,
        const real &density,
        const real &nu_tilde,
        const real &d_wall,
        const real &s_tilde) const;

    /// Source term with cross derivatives for the negative SA model
    /** Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     *  eq.(17) for positive nu_tilde
     */ 
    std::array<real,nstate> compute_cross_source (
        const real &density,
        const real &nu_tilde,
        const real &laminar_kinematic_viscosity,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<dealii::Tensor<1,dim,real>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const;

    /** Constant coefficients for the negative SA model
     *  Reference: Steven R. Allmaras. (2012). "Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model."
     */ 
    const real c_b1  = 0.1355;
    const real sigma = 2.0/3.0;
    const real c_b2  = 0.622;
    const real kappa = 0.41;
    const real c_w1  = c_b1/(kappa*kappa)+(1+c_b2)/sigma;
    const real c_w2  = 0.3;
    const real c_w3  = 2.0;
    const real c_v1  = 7.1;
    const real c_v2  = 0.7;
    const real c_v3  = 0.9;
    const real c_t3  = 1.2;
    const real c_t4  = 0.5;
    const real c_n1  = 16.0;
    const real r_lim = 10.0;

    const FadType sigma_fad = 2.0/3.0;
    const FadType c_v1_fad  = 7.1;
    const FadType c_n1_fad  = 16.0;
};


} // Physics namespace
} // PHiLiP namespace

#endif