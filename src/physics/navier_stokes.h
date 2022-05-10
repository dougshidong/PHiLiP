#ifndef __NAVIER_STOKES__
#define __NAVIER_STOKES__

#include "euler.h"
#include "parameters/parameters_navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Navier-Stokes equations. Derived from Euler for the convective terms, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class NavierStokes : public Euler <dim, nstate, real>
{
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
	/// Constructor
	NavierStokes( 
	    const double                                              ref_length,
	    const double                                              gamma_gas,
	    const double                                              mach_inf,
	    const double                                              angle_of_attack,
	    const double                                              side_slip_angle,
	    const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic,
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
	    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

	/// Nondimensionalized viscosity coefficient at infinity.
	const double viscosity_coefficient_inf;
	/// Prandtl number
	const double prandtl_number;
	/// Farfield (free stream) Reynolds number
	const double reynolds_number_inf;
    /// Nondimensionalized isothermal wall temperature
    const double isothermal_wall_temperature;
    /// Thermal boundary condition type (adiabatic or isothermal)
    const thermal_boundary_condition_enum thermal_boundary_condition_type;

    /** Obtain gradient of primitive variables from gradient of conservative variables */
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    convert_conservative_gradient_to_primitive_gradient (
    	const std::array<real2,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized temperature gradient */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_temperature_gradient (
    	const std::array<real2,nstate> &primitive_soln,
    	const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     *  Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    template<typename real2>
    real2 compute_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized viscosity coefficient, hat{mu*} 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    template<typename real2>
    real2 compute_scaled_viscosity_coefficient (const std::array<real2,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized heat conductivity, hat{kappa*}
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    real2 compute_scaled_heat_conductivity (const std::array<real2,nstate> &primitive_soln) const;

    /** Nondimensionalized heat flux, q*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_heat_flux (
    	const std::array<real2,nstate> &primitive_soln,
    	const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Evaluate vorticity from conservative variables and gradient of conservative variables
    dealii::Tensor<1,3,real> compute_vorticity (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate enstrophy from conservative variables and gradient of conservative variables
    real compute_enstrophy (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate non-dimensional theoretical vorticity-based dissipation rate integrated enstrophy. 
     *  Note: For incompressible flows or when dilatation effects are negligible
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     *  -- Equation (56) with free-stream nondimensionalization applied
     * */
    real compute_vorticity_based_dissipation_rate_from_integrated_enstrophy (
        const real integrated_enstrophy) const;

    /** Evaluate pressure dilatation from conservative variables and gradient of conservative variables
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    real compute_pressure_dilatation (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate the deviatoric strain-rate tensor from conservative variables and gradient of conservative variables
     *  -- Reference: Pope, Stephen B., "Turbulent Flows", Cambridge University Press (2000). Eq.(2.70)
     * */
    dealii::Tensor<2,dim,real> compute_deviatoric_strain_rate_tensor (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /// Evaluate the square of the deviatoric strain-rate tensor magnitude (i.e. double dot product) from conservative variables and gradient of conservative variables
    real compute_deviatoric_strain_rate_tensor_magnitude_sqr (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Evaluate non-dimensional theoretical deviatoric strain-rate tensor based dissipation rate from integrated
     *  deviatoric strain-rate tensor magnitude squared.
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     *  -- Equation (57a) with free-stream nondimensionalization applied
     * */
    real compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr (
        const real integrated_deviatoric_strain_rate_tensor_magnitude_sqr) const;

    /** Extract gradient of velocities */
    template<typename real2>
    dealii::Tensor<2,dim,real2> 
    extract_velocities_gradient_from_primitive_solution_gradient (
    	const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    template<typename real2>
    dealii::Tensor<2,dim,real2>
    compute_viscous_stress_tensor (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
	std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const override;

	/** Gradient of the scaled nondimensionalized viscosity coefficient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14 and 4.14.17)
     */
	dealii::Tensor<1,dim,real> compute_scaled_viscosity_gradient (
    	const std::array<real,nstate> &primitive_soln,
    	const dealii::Tensor<1,dim,real> temperature_gradient) const;

	/** Dissipative flux Jacobian 
	 *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
	 */
	dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal) const;

    /** Dissipative flux Jacobian wrt gradient component
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient) const;

    /// Dissipative flux contribution to the source term
    std::array<real,nstate> dissipative_source_term (
        const dealii::Point<dim,real> &pos) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real /*current_time*/) const override;

    /// Convective flux Jacobian computed via dfad (automatic differentiation)
    /// -- Only used for verifying the dfad procedure used in dissipative flux jacobian
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian_via_dfad (
        std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Scaled viscosity coefficient derivative wrt temperature via dfad (automatic differentiation)
    /// -- Only used for verifying the basic dfad procedure that is extended in convective_flux_directional_jacobian_via_dfad()
    real compute_scaled_viscosity_coefficient_derivative_wrt_temperature_via_dfad (
        std::array<real,nstate> &conservative_soln) const;

protected:    
    ///@{
    /** Constants for Sutherland's law for viscosity
     *  Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     *  Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const double free_stream_temperature = 273.15; ///< Free stream temperature. Units: [K]
    const double sutherlands_temperature = 110.4; ///< Sutherland's temperature. Units: [K]
    const double temperature_ratio = sutherlands_temperature/free_stream_temperature;
    //@}

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /** No-slip wall boundary conditions
     *  * Given by equations 460-461 of the following paper:
     *  * * Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    void boundary_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

    /// Evaluate the manufactured solution boundary conditions.
    void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

private:
    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    real get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real> &tensor) const;

};

} // Physics namespace
} // PHiLiP namespace

#endif
