#ifndef __ENTROPY_CONS_NUM_FLUX__
#define __ENTROPY_CONS_NUM_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
#include "numerical_flux/convective_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {

/// Entropy Conserving Numerica Flux currently only for Burgers' split-form 1D.
template<int dim, int nstate, typename real>
class EntropyConsNumFlux: public NumericalFluxConvective<dim, nstate, real>
{
public:

/// Constructor
EntropyConsNumFlux(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
pde_physics(physics_input)
{};
/// Destructor
~EntropyConsNumFlux() {};

/// Returns the convective numerical flux at an interface.
std::array<real, nstate> evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal1) const;

protected:
/// Numerical flux requires physics to evaluate convective eigenvalues.
const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;

};

}
}

#endif
