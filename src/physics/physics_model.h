#ifndef __PHYSICS_MODEL__
#define __PHYSICS_MODEL__

/// Files for the baseline physics
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Physics Model equations. Derived from PhysicsBase, holds a baseline physics and model terms and equations. 
template <int dim, int nstate, typename real>
class PhysicsModelBase : public PhysicsBase <dim, nstate, real>
{
public:
	
    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
