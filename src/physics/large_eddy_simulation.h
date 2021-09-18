#ifndef __NAVIER_STOKES__
#define __LARGE_EDDY_SIMULATION__

#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class LargeEddySimulationBase : public NavierStokes <dim, nstate, real>
{
public:
	/// Constructor
	LargeEddySimulationBase( 
	    const double                                              ref_length,
	    const double                                              gamma_gas,
	    const double                                              mach_inf,
	    const double                                              angle_of_attack,
	    const double                                              side_slip_angle,
	    const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
	    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

	/// Turbulent Prandtl number
	const double turbulent_prandtl_number;

    /// Nondimensionalized heat flux, q*
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_heat_flux (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const override;

    /// Nondimensionalized viscous stress tensor, tau*
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,dim> 
    compute_viscous_stress_tensor (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const override;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    virtual template<typename real2> std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_stress_tensor (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    virtual template<typename real2> dealii::Tensor<1,dim,real2> compute_SGS_heat_flux (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const = 0;
};

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class LargeEddySimulation_Smagorinsky : public LargeEddySimulationBase <dim, nstate, real>
{
public:
    /// Constructor -- NOT DONE
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: To be put here
     */
    LargeEddySimulation_Smagorinsky( 
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const double                                              turbulent_prandtl_number,
        const double                                              model_constant,
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    template<typename real2> 
    std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_stress_tensor (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    template<typename real2> 
    std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_heat_flux (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;

    /// Eddy viscosity
    template<typename real2> real2 compute_eddy_viscosity(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const;
};
// test commit
} // Physics namespace
} // PHiLiP namespace

#endif
