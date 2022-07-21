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
        //const double                                              ratio_of_filter_width_to_cell_size,
        //no need for RANS models
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

    /// Destructor
    ~ReynoldsAveragedNavierStokesBase
() {};

    /// Turbulent Prandtl number
    const double turbulent_prandtl_number;

    /// Ratio of filter width to cell size
    //const double ratio_of_filter_width_to_cell_size;
    //no need for RANS models

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,nstate,real> > navier_stokes_physics;

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

    /// Compute the nondimensionalized filter width used by the SGS model given a cell index
    //double get_filter_width (const dealii::types::global_dof_index cell_index) const;
    //no need for RANS models

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    virtual dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*
    virtual dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<1,dim,FadType> compute_Reynolds_heat_flux_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

protected:
    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real2> &tensor) const;

    /// Templated dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

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
        //const double                                              ratio_of_filter_width_to_cell_size,
        //no need for RANS models
        //const double                                              model_constant,
        //no need for RANS models
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

    /// SGS model constant
    //const double model_constant;
    //no need for RANS models

    /// Destructor
    ~ReynoldsAveragedNavierStokes_SAneg() {};

    /// Returns the product of the eddy viscosity model constant and the filter width
    //double get_model_constant_times_filter_width (const dealii::types::global_dof_index cell_index) const;
    //no need for RANS models

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*
    dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType)
    dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType)
    dealii::Tensor<1,dim,FadType> compute_Reynolds_heat_flux_fad (
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized eddy viscosity for the negative SA model
    virtual real compute_eddy_viscosity(
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Nondimensionalized eddy viscosity for the negative SA model (Automatic Differentiation Type: FadType)
    virtual FadType compute_eddy_viscosity_fad(
        const std::array<FadType,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

protected:
    /// Templated nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    template<typename real2> dealii::Tensor<2,dim,real2> compute_Reynolds_stress_tensor_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Templated nondimensionalized Reynolds heat flux, (q^reynolds)*
    template<typename real2> 
    dealii::Tensor<1,dim,real2> compute_Reynolds_heat_flux_templated (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Templated scale nondimensionalized eddy viscosity for the negative SA model
    template<typename real2> real2 scale_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const real2 eddy_viscosity) const;

private:
    /// Templated nondimensionalized eddy viscosity for the negative SA model.
    template<typename real2> real2 compute_eddy_viscosity_templated(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
        const dealii::types::global_dof_index cell_index) const;
};


} // Physics namespace
} // PHiLiP namespace

#endif