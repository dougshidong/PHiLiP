#ifndef __PHYSICS_MODEL__
#define __PHYSICS_MODEL__

#include "physics/navier_stokes.h"
#include "parameters/parameters_physics_model.h"
#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace PhysicsModel {

/// Physics model additional terms and equations to the baseline physics. 
template <int dim, int nstate, typename real>
class PhysicsModelBase
{
public:
	/// Constructor
	PhysicsModelBase();

    /// Model convective flux terms additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    model_convective_flux (
        const std::array<real,nstate> &conservative_soln) const = 0;

    /// Model dissipative flux terms additional to the baseline physics
	virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
	model_dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

    /// Model source terms additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    model_source_term (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;
};

} // PhysicsModel namespace
} // PHiLiP namespace

#endif
