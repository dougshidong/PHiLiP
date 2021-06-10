#ifndef __NAVIER_STOKES__
#define __NAVIER_STOKES__

#include "euler.h"

namespace PHiLiP {
namespace Physics {

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
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    convert_conservative_gradient_to_primitive_gradient (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized temperature gradient */
    dealii::Tensor<1,dim,real> compute_temperature_gradient (
    	const std::array<real,nstate> &primitive_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscosity coefficient, $\mu^{*}$ 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     *  Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    real compute_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$ 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    real compute_scaled_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$ 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    real compute_scaled_heat_conductivity (const std::array<real,nstate> &primitive_soln) const;

    /** Nondimensionalized heat flux, $\bm{q}^{*}$ 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,real> compute_heat_flux (
    	const std::array<real,nstate> &primitive_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Extract gradient of velocities */
    std::array<dealii::Tensor<1,dim,real>,dim> 
    extract_velocities_gradient_from_primitive_solution_gradient (
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    std::array<dealii::Tensor<1,dim,real>,dim> 
    compute_viscous_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
	std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const override;
	/// Here, dissipative_flux() is the only virtual from Euler

	/** Gradient of the scaled nondimensionalized viscosity coefficient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14 and 4.14.17)
     */
	dealii::Tensor<1,dim,real> compute_scaled_viscosity_gradient (
    	const std::array<real,nstate> &primitive_soln,
    	const dealii::Tensor<1,dim,real> temperature_gradient) const;

	/** Dissipative flux Jacobian 
	 *  Reference: Masatsuka 2018 "I do like CFD", p.?
	 */
	// dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
 //    	const std::array<real,nstate> &conservative_soln,
 //    	const dealii::Tensor<1,dim,real> &normal) const;


	/// Source term is zero or depends on manufactured solution
	/// Not a virtual in euler.hpp --> Must change it to virtual class
	/// --- do we need to specify override??
    // std::array<real,nstate> source_term (
    //     const dealii::Point<dim,real> &pos,
    //     const std::array<real,nstate> &conservative_soln) const override;
protected:
    /** Constants for Sutherland's law for viscosity
     *  Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     *  Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const double free_stream_temperature = 273.15; // Free stream temperature. Units: [K]
    const double sutherlands_temperature = 110.4; // Sutherland's temperature. Units: [K]
    const double temperature_ratio = sutherlands_temperature/free_stream_temperature;
};

} // Physics namespace
} // PHiLiP namespace

#endif
