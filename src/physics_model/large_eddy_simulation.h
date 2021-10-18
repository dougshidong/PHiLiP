#ifndef __LARGE_EDDY_SIMULATION__
#define __LARGE_EDDY_SIMULATION__

#include "physics_model.h"

namespace PHiLiP {

/// Physics Model equations. Derived from PhysicsBase, holds a baseline physics and model terms and equations. 
template <int dim, int nstate, typename real>
class LargeEddySimulationBase : public PhysicsModelBase <dim, nstate, real>
{
public:
	
    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> model_convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> model_dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

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

} // PHiLiP namespace

#endif
