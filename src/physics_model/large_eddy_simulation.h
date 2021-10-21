#ifndef __LARGE_EDDY_SIMULATION__
#define __LARGE_EDDY_SIMULATION__

#include "physics_model.h"

namespace PHiLiP {

/// Large Eddy Simulation equations. Derived from Navier-Stokes for modifying the stress tensor and heat flux, which is derived from PhysicsBase. 
template <int dim, int nstate, typename real>
class LargeEddySimulationBase : public PhysicsModelBase <dim, nstate, real>
{
public:
    /// Constructor
	LargeEddySimulationBase( 
	    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
        const double                                                     turbulent_prandtl_number);

    /// Navier-Stokes physics object
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics;

	/// Turbulent Prandtl number
	const double turbulent_prandtl_number;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> model_convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> model_dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Nondimensionalized sub-grid scale (SGS) stress tensor, (tau^sgs)*
    virtual template<typename real2> std::array<dealii::Tensor<1,dim,real2>,dim> compute_SGS_stress_tensor (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const = 0;

    /// Nondimensionalized sub-grid scale (SGS) heat flux, (q^sgs)*
    virtual template<typename real2> dealii::Tensor<1,dim,real2> compute_SGS_heat_flux (
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const = 0;

protected:
    /// Returns the square of the magnitude of the tensor
    template<typename real2> 
    real2 get_tensor_magnitude_sqr (const std::array<dealii::Tensor<1,dim,real2>,dim> &tensor) const;
};

/// Smagorinsky eddy viscosity model. Derived from Large Eddy Simulation.
template <int dim, int nstate, typename real>
class LargeEddySimulation_Smagorinsky : public LargeEddySimulationBase <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: To be put here
     */
    LargeEddySimulation_Smagorinsky( 
        const double                                                     model_constant,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
        const double                                                     turbulent_prandtl_number);

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

/// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model. Derived from LargeEddySimulation_Smagorinsky for only modifying compute_eddy_viscosity.
template <int dim, int nstate, typename real>
class LargeEddySimulation_WALE : public LargeEddySimulation_Smagorinsky <dim, nstate, real>
{
public:
    /** Constructor for the sub-grid scale (SGS) model: Smagorinsky
     *  More details...
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    LargeEddySimulation_WALE( 
        const double                                                     model_constant,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,dim+2,real> >  navier_stokes_physics_input,
        const double                                                     turbulent_prandtl_number);

    /** Eddy viscosity for the WALE model. 
     *  Reference: Nicoud & Ducros (1999) "Subgrid-scale stress modelling based on the square of the velocity gradient tensor"
     */
    template<typename real2> real2 compute_eddy_viscosity(
        const std::array<real2,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient) const override;
};

} // PHiLiP namespace

#endif
