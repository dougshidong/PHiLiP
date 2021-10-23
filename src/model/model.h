#ifndef __MODEL__
#define __MODEL__

#include "physics/navier_stokes.h"
#include "parameters/parameters_physics_model.h"
#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace Physics {

/// Physics model additional terms and equations to the baseline physics. 
template <int dim, int nstate, typename real>
class ModelBase
{
public:
	/// Constructor
	ModelBase();

    /// Convective flux terms additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    convective_flux (
        const std::array<real,nstate> &conservative_soln) const = 0;

    /// Dissipative flux terms additional to the baseline physics
	virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

    /// Source terms additional to the baseline physics
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const = 0;
};

} // Physics namespace
} // PHiLiP namespace

#endif
