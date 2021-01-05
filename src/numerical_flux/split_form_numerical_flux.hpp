#ifndef __SPLIT_NUM_FLUX__
#define __SPLIT_NUM_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
#include "numerical_flux/convective_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {

/// Lax-Friedrichs numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class SplitFormNumFlux: public NumericalFluxConvective<dim, nstate, real>
{
public:

/// Constructor
SplitFormNumFlux(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
pde_physics(physics_input)
{};
/// Destructor
~SplitFormNumFlux() {};

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
