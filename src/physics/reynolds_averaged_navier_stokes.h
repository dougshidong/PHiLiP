#ifndef __REYNOLDS_AVERAGED_NAVIER_STOKES__
#define __REYNOLDS_AVERAGED_NAVIER_STOKES__

#include "model.h"
#include "navier_stokes.h"
#include "euler.h"

namespace PHiLiP {
namespace Physics {

/// Reynolds-Averaged Navier-Stokes equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class ReynoldsAveragedNavierStokesBase : public ModelBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    /// Constructor
	ReynoldsAveragedNavierStokesBase
(
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
    ~ReynoldsAveragedNavierStokesBase
() {};

    /// Turbulent Prandtl number
    const double turbulent_prandtl_number;

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,dim+2,real> > navier_stokes_physics;

    /// Additional convective flux of RANS + convective flux of turbulence model
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Additional viscous flux of RANS + viscous flux of turbulence model
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Convective eigenvalues
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective and dissipative source term for manufactured solution functions
    std::array<real,nstate> convective_dissipative_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const dealii::types::global_dof_index cell_index) const;

    /// Physical source term
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    virtual dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*
    virtual dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<1,dim,FadType> compute_Reynolds_heat_flux_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized effective (total) viscosities for the turbulence model
    virtual std::array<real,nstate-(dim+2)> compute_effective_viscosity_turbulence_model (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized effective (total) viscosities for the turbulence model (Automatic Differentiation Type: FadType)
    virtual std::array<FadType,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Physical source term (production, dissipation source terms and source term with cross derivatives) in the turbulence model
    virtual std::array<real,nstate> compute_production_dissipation_cross_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

protected:
    /// Returns the square of the magnitude of the vector 
    template<typename real2> 
    real2 get_vector_magnitude_sqr (const dealii::Tensor<1,3,real2> &vector) const;

    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real2> &tensor) const;

    /// Templated additional dissipative (i.e. viscous) flux
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Templated additional convective flux
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> convective_flux_templated (
        const std::array<real2,nstate> &conservative_soln) const;

    /// Returns the conservative solutions of Reynolds-averaged Navier-Stokes equations
    template <typename real2>
    std::array<real2,dim+2> extract_rans_conservative_solution (
        const std::array<real2,nstate> &conservative_soln) const;

    /// Returns the conservative solutions gradient of Reynolds-averaged Navier-Stokes equations
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim+2> extract_rans_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /// Templated Additional viscous flux of RANS + viscous flux of turbulence model
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> dissipative_flux_turbulence_model (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model,
        const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const;

    /// Given conservative variables of turbulence model
    /// Return primitive variables of turbulence model
    template <typename real2>
    std::array<real2,nstate-(dim+2)> convert_conservative_to_primitive_turbulence_model (
        const std::array<real2,nstate> &conservative_soln) const;

    /// Given conservative variable gradients of turbulence model
    /// Return primitive variable gradients of turbulence model
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> convert_conservative_gradient_to_primitive_gradient_turbulence_model (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /// Mean turbulence properties given two sets of conservative solutions
    /** Used in the implementation of the split form.
     */
    std::array<real,nstate-(dim+2)> compute_mean_turbulence_property (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /** convective flux Jacobian 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /** Dissipative flux Jacobian 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian wrt gradient component 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Get manufactured solution value 
    std::array<real,nstate> get_manufactured_solution_value (
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution value 
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient (
        const dealii::Point<dim,real> &pos) const;

    /** Convective flux contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    std::array<real,nstate> convective_source_term (
        const dealii::Point<dim,real> &pos) const;

    /** Dissipative flux contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    std::array<real,nstate> dissipative_source_term (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    /** Physical source contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     */
    std::array<real,nstate> physical_source_source_term (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;
};

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
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated nondimensionalized effective (total) viscosities for the negative SA model
    template<typename real2> std::array<real2,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated coefficient Chi for the negative SA model
    template<typename real2> real2 compute_coefficient_Chi (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Templated coefficient f_v1 for the negative SA model
    template<typename real2> real2 compute_coefficient_f_v1 (
        const real2 &coefficient_Chi) const;

    /// Coefficient f_v2 for the negative SA model.
    real compute_coefficient_f_v2 (
        const real &coefficient_Chi) const;

    /// Templated coefficient f_n for the negative SA model
    template<typename real2> real2 compute_coefficient_f_n (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Coefficient f_t2 for the negative SA model
    real compute_coefficient_f_t2 (
        const real &coefficient_Chi) const;

    /// Coefficient f_t2 for the negative SA model
    real compute_coefficient_f_w (
        const real &coefficient_g) const;

    /// Coefficient r for the negative SA model
    real compute_coefficient_r (
        const real &nu_tilde,
        const real &d_wall,
        const real &s_tilde) const;

    /// Coefficient g for the negative SA model
    real compute_coefficient_g (
        const real &coefficient_r) const;

    /// Vorticity magnitude for the negative SA model
    real compute_s (
        const std::array<real,dim+2> &conservative_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &conservative_soln_gradient_rans) const;

    /// Correction of vorticity magnitude for the negative SA model
    real compute_s_bar (
        const real &coefficient_Chi,
        const real &nu_tilde,
        const real &d_wall) const;

    /// Modified vorticity magnitude for the negative SA model
    real compute_s_tilde (
        const real &coefficient_Chi,
        const real &nu_tilde,
        const real &d_wall,
        const real &s) const;

    /// Production source term for the negative SA model
    std::array<real,nstate> compute_production_source (
        const real &coefficient_f_t2,
        const real &density,
        const real &nu_tilde,
        const real &s,
        const real &s_tilde) const;

    /// Dissipation source term for the negative SA model
    std::array<real,nstate> compute_dissipation_source (
        const real &coefficient_f_t2,
        const real &density,
        const real &nu_tilde,
        const real &d_wall,
        const real &s_tilde) const;

    /// Source term with cross derivatives for the negative SA model
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