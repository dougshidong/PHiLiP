#ifndef __REYNOLDS_AVERAGED_NAVIER_STOKES__
#define __REYNOLDS_AVERAGED_NAVIER_STOKES__

#include "model.h"
#include "navier_stokes.h"
#include "euler.h"

namespace PHiLiP {
namespace Physics {

/// Reynolds-Averaged Navier-Stokes (RANS) equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class ReynoldsAveragedNavierStokesBase : public ModelBase <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
	ReynoldsAveragedNavierStokesBase(
	    const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG);

    /// Destructor
    ~ReynoldsAveragedNavierStokesBase() {};

    /// Number of PDEs for RANS equations
    static const int nstate_navier_stokes = dim+2;

    /// Number of PDEs for RANS turbulence model
    static const int nstate_turbulence_model = nstate-(dim+2);

    /// Turbulent Prandtl number
    const double turbulent_prandtl_number;

    /// Pointer to Navier-Stokes physics object
    std::unique_ptr< NavierStokes<dim,nstate_navier_stokes,real> > navier_stokes_physics;

    /// Additional convective flux of RANS + convective flux of turbulence model
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Additional viscous flux of RANS + viscous flux of turbulence model
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective eigenvalues of the additional models' PDEs
    /** For RANS model, all entries associated with RANS are assigned to be zero 
     *                  all entries associated with turbulence model are assigned to be the corresponding eigenvalues*/
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue of the additional models' PDEs
    /** For RANS model, this value is assigned to be the maximum eigenvalue of the turbulence model 
     *  In most of the RANS models, the maximum eigenvalue of the convective flux is the flow velocity */
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_solution,
        const real current_time,
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
        const dealii::types::global_dof_index cell_index) const override;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)*
    virtual dealii::Tensor<2,dim,real> compute_Reynolds_stress_tensor (
        const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &primitive_soln_gradient_rans,
        const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)*
    virtual dealii::Tensor<1,dim,real> compute_Reynolds_heat_flux (
        const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,real>,nstate_navier_stokes> &primitive_soln_gradient_rans,
        const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds stress tensor, (tau^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<2,dim,FadType> compute_Reynolds_stress_tensor_fad (
        const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate_navier_stokes> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized Reynolds heat flux, (q^reynolds)* (Automatic Differentiation Type: FadType)
    virtual dealii::Tensor<1,dim,FadType> compute_Reynolds_heat_flux_fad (
        const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<dealii::Tensor<1,dim,FadType>,nstate_navier_stokes> &primitive_soln_gradient_rans,
        const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized effective (total) viscosities for the turbulence model
    virtual std::array<real,nstate_turbulence_model> compute_effective_viscosity_turbulence_model (
        const std::array<real,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<real,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

    /// Nondimensionalized effective (total) viscosities for the turbulence model (Automatic Differentiation Type: FadType)
    virtual std::array<FadType,nstate_turbulence_model> compute_effective_viscosity_turbulence_model_fad (
        const std::array<FadType,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<FadType,nstate_turbulence_model> &primitive_soln_turbulence_model) const = 0;

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

    /// Returns the conservative solutions of Reynolds-averaged Navier-Stokes equations (without additional RANS turbulence model)
    template <typename real2>
    std::array<real2,dim+2> extract_rans_conservative_solution (
        const std::array<real2,nstate> &conservative_soln) const;

    /// Returns the conservative solutions gradient of Reynolds-averaged Navier-Stokes equations (without additional RANS turbulence model)
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim+2> extract_rans_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /// Templated Additional viscous flux of RANS + viscous flux of turbulence model
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> dissipative_flux_turbulence_model (
        const std::array<real2,nstate_navier_stokes> &primitive_soln_rans,
        const std::array<real2,nstate_turbulence_model> &primitive_soln_turbulence_model,
        const std::array<dealii::Tensor<1,dim,real2>,nstate_turbulence_model> &primitive_solution_gradient_turbulence_model) const;

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

    /** Convective flux Jacobian 
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
    std::array<real,nstate> convective_source_term_computed_from_manufactured_solution (
        const dealii::Point<dim,real> &pos) const;

    /** Dissipative flux contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    std::array<real,nstate> dissipative_source_term_computed_from_manufactured_solution (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    /** Physical source contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     */
    std::array<real,nstate> physical_source_term_computed_from_manufactured_solution (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    /// Evaluate the manufactured solution boundary conditions.
    void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;
};

} // Physics namespace
} // PHiLiP namespace

#endif