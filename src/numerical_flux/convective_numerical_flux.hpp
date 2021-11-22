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

/// Base class of Roe (Roe-Pike) flux with entropy fix. Derived from NumericalFluxConvective.
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

	/// Destructor
	~RoeBase() {};

	/// Virtual member function for evaluating the entropy fix for a Roe-Pike flux.
	virtual void evaluate_entropy_fix (
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const = 0;

	/// Virtual member function for evaluating additional modifications/corrections for a Roe-Pike flux.
	virtual void evaluate_additional_modifications (
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const = 0;

	/// Returns the convective flux at an interface
	/// --- See Blazek 2015, p.103-105
	/// --- Note: Modified calculation of alpha_{3,4} to use 
	///           dVt (jump in tangential velocities);
	///           expressions are equivalent.
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

	/// Evaluates the entropy fix of Harten
	/// --- See Blazek 2015, p.103-105
	void evaluate_entropy_fix(
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const;

	/// Empty function. No additional modifications for the Roe-Pike scheme.
	void evaluate_additional_modifications(
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const;
};

/// L2Roe flux with entropy fix. Derived from RoeBase.
/// --- Reference: Osswald et al. (2016 L2Roe)
template<int dim, int nstate, typename real>
class L2Roe: public RoeBase<dim, nstate, real>
{
public:
	/// Constructor
	L2Roe(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
		:	RoeBase<dim, nstate, real>(physics_input){}

	/// (1) Van Leer et al. (1989 Sonic) entropy fix for acoustic waves (i.e. i=1,5)
	/// (2) For waves (i=2,3,4) --> Entropy fix of Liou (2000 Mass)
	/// --- See p.74 of Osswald et al. (2016 L2Roe)
	void evaluate_entropy_fix(
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    std::array<real, 3> &eig_RoeAvg,
	    const real vel2_ravg,
	    const real sound_ravg) const;
	
	/// Osswald's two modifications to Roe-Pike scheme --> L2Roe
	/// --- Scale jump in (1) normal and (2) tangential velocities using a blending factor
	void evaluate_additional_modifications(
	    const std::array<real, nstate> &soln_int,
	    const std::array<real, nstate> &soln_ext,
	    const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    real &dV_normal, 
	    dealii::Tensor<1,dim,real> &dV_tangent) const;

protected:
	/// Shock indicator of Wada & Liou (1994 Flux) -- Eq.(39)
	/// --- See also p.74 of Osswald et al. (2016 L2Roe)
	void evaluate_shock_indicator(
		const std::array<real, 3> &eig_L,
	    const std::array<real, 3> &eig_R,
	    int &ssw_LEFT,
	    int &ssw_RIGHT) const;
};

#if 0
/// Central numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class CentralFlux: public NumericalFluxConvective<dim, nstate, real>
{
public:

/// Constructor
CentralFlux(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
pde_physics(physics_input)
{};
/// Destructor
~CentralFlux() {};

/// Returns the Central convective numerical flux at an interface.
std::array<real, nstate> evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal1) const;

protected:
/// Numerical flux requires physics to evaluate convective eigenvalues.
const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;

};
#endif

} /// NumericalFlux namespace
} /// PHiLiP namespace

#endif
