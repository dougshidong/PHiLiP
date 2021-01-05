#ifndef __NUMERICAL_FLUX__
#define __NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "numerical_flux/viscous_numerical_flux.h"
#include "physics/physics.h"
#include "physics/euler.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

/// Base class of numerical flux associated with convection
template<int dim, int nstate, typename real>
class NumericalFluxConvective
{
public:
virtual ~NumericalFluxConvective() = 0; ///< Base class destructor required for abstract classes.

/// Returns the convective numerical flux at an interface.
virtual std::array<real, nstate> evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal1) const = 0;

};


/// Lax-Friedrichs numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class LaxFriedrichs: public NumericalFluxConvective<dim, nstate, real>
{
public:

/// Constructor
LaxFriedrichs(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
pde_physics(physics_input)
{};
/// Destructor
~LaxFriedrichs() {};

/// Returns the Lax-Friedrichs convective numerical flux at an interface.
std::array<real, nstate> evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal1) const;

protected:
/// Numerical flux requires physics to evaluate convective eigenvalues.
const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;

};

/// Roe flux with entropy fix. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class Roe: public NumericalFluxConvective<dim, nstate, real>
{
public:

/// Constructor
Roe(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
euler_physics(std::dynamic_pointer_cast<Physics::Euler<dim,nstate,real>>(physics_input))
{};
/// Destructor
~Roe() {};

/// Returns the Roe convective numerical flux at an interface.
std::array<real, nstate> evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal1) const;

protected:
/// Numerical flux requires physics to evaluate convective eigenvalues.
const std::shared_ptr < Physics::Euler<dim, nstate, real> > euler_physics;

};


} // NumericalFlux namespace
} // PHiLiP namespace

#endif
