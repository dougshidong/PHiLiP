#ifndef __NUMERICAL_FLUX__
#define __NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
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

/// Base class of Roe flux with entropy fix. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class RoeBase: public NumericalFluxConvective<dim, nstate, real>
{
protected:
	/// Numerical flux requires physics to evaluate convective eigenvalues.
	const std::shared_ptr < Physics::Euler<dim, nstate, real> > euler_physics;

public:
	/// Constructor
	RoeBase(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
	:
	euler_physics(std::dynamic_pointer_cast<Physics::Euler<dim,nstate,real>>(physics_input))
	{};

    /// Virtual destructor required for abstract classes.
	~RoeBase() {};

	virtual void evaluate_entropy_fix (
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const = 0;

	virtual void evaluate_additional_modifications (
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const = 0;

	/// Returns the convective flux at an interface
	std::array<real, nstate> evaluate_flux (
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const dealii::Tensor<1,dim,real> &normal1) const;
};

/// RoePike flux with entropy fix. Derived from RoeBase.
template<int dim, int nstate, typename real>
class RoePike: public RoeBase<dim, nstate, real>
{
public:
	/// Constructor
	RoePike(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
		:	RoeBase<dim, nstate, real>(physics_input){}

	void evaluate_entropy_fix(
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const;

	void evaluate_additional_modifications(
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const;
};

/// L2Roe flux with entropy fix. Derived from RoeBase.
template<int dim, int nstate, typename real>
class L2Roe: public RoeBase<dim, nstate, real>
{
public:
	/// Constructor
	L2Roe(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
		:	RoeBase<dim, nstate, real>(physics_input){}

	void evaluate_entropy_fix(
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const;
	
	void evaluate_additional_modifications(
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const;

protected:
	void evaluate_shock_indicator(
		const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    int &ssw_LEFT,
	    int &ssw_RIGHT) const;
};


} /// NumericalFlux namespace
} /// PHiLiP namespace

#endif
