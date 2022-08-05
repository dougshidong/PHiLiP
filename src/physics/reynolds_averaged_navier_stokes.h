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

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
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

    /// Nondimensionalized eddy viscosity for the negative SA model
    virtual real compute_eddy_viscosity(
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized eddy viscosity for the negative SA model (Automatic Differentiation Type: FadType)
    virtual FadType compute_eddy_viscosity_fad(
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    virtual std::array<real,nstate-(dim+2)> compute_effective_viscosity_turbulence_model (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

    virtual std::array<FadType,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const = 0;

protected:
    /// Returns the square of the magnitude of the vector (i.e. the double dot product of a vector with itself)
    template<typename real2> 
    real2 get_vector_magnitude_sqr (const dealii::Tensor<1,dim,real2> &vector) const;

    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real2> &tensor) const;

    /// Templated dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    template <typename real2>
    std::array<real2,dim+2> extract_rans_conservative_solution (
        const std::array<real2,nstate> &conservative_soln) const;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim+2> extract_rans_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

/*
    template <typename real2>
    std::array<real2,nstate-(dim+2)> extract_turbulence_model_conservative_solution (
        const std::array<real2,nstate> &conservative_soln) const;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> extract_turbulence_model_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;
*/

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> dissipative_flux_turbulence_model (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model,
        const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const;

    template <typename real2>
    std::array<real2,nstate-(dim+2)> convert_conservative_to_primitive_turbulence_model (
        const std::array<real2,nstate> &conservative_soln) const;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> convert_conservative_gradient_to_primitive_gradient_turbulence_model (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /** Dissipative flux Jacobian (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    //not sure if it is needed for RANS
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian wrt gradient component (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    //not sure if it is needed for RANS
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Get manufactured solution value (repeated from Euler)
    //not sure if it is needed for RANS
    std::array<real,nstate> get_manufactured_solution_value (
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution value (repeated from Euler)
    //not sure if it is needed for RANS
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient (
        const dealii::Point<dim,real> &pos) const;

    /// Dissipative flux contribution to the source term (repeated from NavierStokes)
    //not sure if it is needed for RANS
    std::array<real,nstate> dissipative_source_term (
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

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*
    dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType)
    dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType)
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

    std::array<real,nstate-(dim+2)> compute_effective_viscosity_turbulence_model (
        const std::array<real,dim+2> &primitive_soln_rans,
        const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    std::array<FadType,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_fad (
        const std::array<FadType,dim+2> &primitive_soln_rans,
        const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const;


protected:
    /// Templated nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    template<typename real2> dealii::Tensor<2,dim,real2> compute_Reynolds_stress_tensor_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated nondimensionalized Reynolds heat flux, (q^reynolds)*
    template<typename real2> 
    dealii::Tensor<1,dim,real2> compute_Reynolds_heat_flux_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated scale nondimensionalized eddy viscosity for the negative SA model
    template<typename real2> real2 scale_eddy_viscosity_templated(
        const std::array<real2,dim+2> &primitive_soln_rans,
        const real2 eddy_viscosity) const;

private:
    /// Templated nondimensionalized eddy viscosity for the negative SA model.
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    template<typename real2> std::array<real2,nstate-(dim+2)> compute_effective_viscosity_turbulence_model_templated (
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const;

    /// Templated coefficient Chi for the negative SA model.
    template<typename real2> real2 compute_coefficient_Chi (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Templated coefficient f_v1 for the negative SA model.
    template<typename real2> real2 compute_coefficient_f_v1 (
        const real2 &coefficient_Chi) const;

    /// Templated coefficient f_v2 for the negative SA model.
    template<typename real2> real2 compute_coefficient_f_v2 (
        const real2 &coefficient_Chi) const;

    /// Templated coefficient f_n for the negative SA model.
    template<typename real2> real2 compute_coefficient_f_n (
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity) const;

    /// Templated coefficient f_t2 for the negative SA model.
    template<typename real2> real2 compute_coefficient_f_t2 (
        const real2 &coefficient_Chi) const;

    /// Templated coefficient f_t2 for the negative SA model.
    template<typename real2> real2 compute_coefficient_f_w (
        const real2 &coefficient_g) const;

    /// Templated coefficient r for the negative SA model.
    template<typename real2> real2 compute_coefficient_r (
        const real2 &nu_tilde,
        const real2 &d_wall,
        const real2 &s_tilde) const;

    /// Templated coefficient g for the negative SA model.
    template<typename real2> real2 compute_coefficient_g (
        const real2 &coefficient_r) const;

    /// Templated vorticity magnitude for the negative SA model.
    template<typename real2> real2 compute_s (
        const std::array<real,dim+2> &conservative_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,dim+2> &conservative_soln_gradient_rans) const;

    /// Templated correction of vorticity magnitude for the negative SA model.
    template<typename real2> real2 compute_s_bar (
        const real2 &coefficient_Chi,
        const real2 &nu_tilde,
        const real2 &d_wall) const;

    /// Templated modified vorticity magnitude for the negative SA model.
    template<typename real2> real2 compute_s_tilde (
        const real2 &coefficient_Chi,
        const real2 &nu_tilde,
        const real2 &d_wall,
        const real2 &s) const;

    template<typename real2> real2 compute_production_source (
        const real2 &coefficient_f_t2,
        const real2 &nu_tilde,
        const real2 &d_wall,
        const real2 &s,
        const real2 &s_tilde) const;

    template<typename real2> real2 compute_dissipation_source (
        const real2 &coefficient_f_t2,
        const real2 &nu_tilde,
        const real2 &d_wall,
        const real2 &s_tilde) const;

    template<typename real2> real2 compute_cross_source (
        const real2 &density,
        const real2 &nu_tilde,
        const real2 &laminar_kinematic_viscosity,
        const std::array<real2,dim+2> &primitive_soln_rans,
        const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model,
        const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
        const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const;

    //constant for negative SA model 
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
    const real c_t1  = 1.0;
    const real c_t2  = 2.0;
    const real c_t3  = 1.2;
    const real c_t4  = 0.5;
    const real c_n1  = 16.0;
    const real r_lim = 10.0;

    const FadType c_b1_fad  = 0.1355;
    const FadType sigma_fad = 2.0/3.0;
    const FadType c_b2_fad  = 0.622;
    const FadType kappa_fad = 0.41;
    const FadType c_w1_fad  = c_b1_fad/(kappa_fad*kappa_fad)+(1+c_b2_fad)/sigma_fad;
    const FadType c_w2_fad  = 0.3;
    const FadType c_w3_fad  = 2.0;
    const FadType c_v1_fad  = 7.1;
    const FadType c_v2_fad  = 0.7;
    const FadType c_v3_fad  = 0.9;
    const FadType c_t1_fad  = 1.0;
    const FadType c_t2_fad  = 2.0;
    const FadType c_t3_fad  = 1.2;
    const FadType c_t4_fad  = 0.5;
    const FadType c_n1_fad  = 16.0;
    const FadType r_lim_fad = 10.0;
};


} // Physics namespace
} // PHiLiP namespace

#endif