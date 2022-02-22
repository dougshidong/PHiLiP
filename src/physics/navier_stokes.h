#ifndef __NAVIER_STOKES__
#define __NAVIER_STOKES__

#include "euler.h"

namespace PHiLiP {
namespace Physics {

/// Navier-Stokes equations. Derived from Euler for the convective terms, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class NavierStokes : public Euler <dim, nstate, real>
{
public:
	/// Constructor
	NavierStokes( 
	    const double                                              ref_length,
	    const double                                              gamma_gas,
	    const double                                              mach_inf,
	    const double                                              angle_of_attack,
	    const double                                              side_slip_angle,
	    const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
	    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

	/// Nondimensionalized viscosity coefficient at infinity.
	const double viscosity_coefficient_inf;
	/// Prandtl number
	const double prandtl_number;
	/// Farfield (free stream) Reynolds number
	const double reynolds_number_inf;

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

    /** Extract gradient of velocities */
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim> 
    extract_velocities_gradient_from_primitive_solution_gradient (
    	const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim> 
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
        const std::array<real,nstate> &conservative_soln,//) const;
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

    /// Boundary face values
    void boundary_face_values (
        const int boundary_type,
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

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

};

} // Physics namespace
} // PHiLiP namespace

#endif
