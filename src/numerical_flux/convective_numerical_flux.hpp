#ifndef __CONVECTIVE_NUMERICAL_FLUX__
#define __CONVECTIVE_NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
#include "physics/euler.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

/// Base class of baseline numerical flux (without upwind term) associated with convection
template<int dim, int nstate, typename real>
class BaselineNumericalFluxConvective
{
public:
    /// Base class destructor required for abstract classes.
    virtual ~BaselineNumericalFluxConvective() = default;

    /// Returns the convective numerical flux at an interface.
    virtual std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const = 0;
};

/// Central numerical flux. Derived from BaselineNumericalFluxConvective.
template<int dim, int nstate, typename real>
class CentralBaselineNumericalFluxConvective : public BaselineNumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit CentralBaselineNumericalFluxConvective(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : pde_physics(physics_input) {};
    
    /// Returns the convective numerical flux at an interface.
    std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;

protected:
    /// Numerical flux requires physics to evaluate convective flux
    const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;
};

/// Entropy Conserving Numerical Flux. Derived from BaselineNumericalFluxConvective.
template<int dim, int nstate, typename real>
class EntropyConservingBaselineNumericalFluxConvective : public BaselineNumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit EntropyConservingBaselineNumericalFluxConvective(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : pde_physics(physics_input) {};
    
    /// Returns the convective numerical flux at an interface.
    std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;

protected:
    /// Numerical flux requires physics to evaluate split form convective flux.
    const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;
};

/// Base class of Riemann solver dissipation (i.e. upwind-term) for numerical flux associated with convection
template<int dim, int nstate, typename real>
class RiemannSolverDissipation
{
public:
    ///< Base class destructor required for abstract classes.
    virtual ~RiemannSolverDissipation() = default;

    /// Returns the convective numerical flux at an interface.
    virtual std::array<real, nstate> evaluate_riemann_solver_dissipation (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const = 0;
};

/// Zero Riemann solver dissipation. Derived from RiemannSolverDissipation.
template<int dim, int nstate, typename real>
class ZeroRiemannSolverDissipation : public RiemannSolverDissipation<dim, nstate, real>
{
public:
    /// Returns zeros
    std::array<real, nstate> evaluate_riemann_solver_dissipation (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;
};

/// Lax-Friedrichs Riemann solver dissipation. Derived from RiemannSolverDissipation.
template<int dim, int nstate, typename real>
class LaxFriedrichsRiemannSolverDissipation : public RiemannSolverDissipation<dim, nstate, real>
{
public:
    /// Constructor
    explicit LaxFriedrichsRiemannSolverDissipation(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : pde_physics(physics_input) {};

    /** Returns the Lax-Friedrichs convective numerical flux at an interface. 
     *  Reference:
     *    Section 3.1 of Bernardo Cockburn, and Chi-Wang Shu, 
     *    "The Runge–Kutta Discontinuous Galerkin Method for Conservation Laws V", 
     *    JOURNAL OF COMPUTATIONAL PHYSICS 141, 199–224 (1998).
     * */
    std::array<real, nstate> evaluate_riemann_solver_dissipation (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;

protected:
    /// Numerical flux requires physics to evaluate convective eigenvalues.
    const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics;
};

/// Base class of Roe (Roe-Pike) flux with entropy fix. Derived from RiemannSolverDissipation.
template<int dim, int nstate, typename real>
class RoeBaseRiemannSolverDissipation : public RiemannSolverDissipation<dim, nstate, real>
{
protected:
    /// Numerical flux requires physics to evaluate convective eigenvalues.
    const std::shared_ptr < Physics::Euler<dim, nstate, real> > euler_physics;

public:
    /// Constructor
    explicit RoeBaseRiemannSolverDissipation(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : euler_physics(std::dynamic_pointer_cast<Physics::Euler<dim,nstate,real>>(physics_input)) {};

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
    std::array<real, nstate> evaluate_riemann_solver_dissipation (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;
};

/// RoePike flux with entropy fix. Derived from RoeBase.
template<int dim, int nstate, typename real>
class RoePikeRiemannSolverDissipation : public RoeBaseRiemannSolverDissipation<dim, nstate, real>
{
public:
    /// Constructor
    explicit RoePikeRiemannSolverDissipation(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : RoeBaseRiemannSolverDissipation<dim, nstate, real>(physics_input){};

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
class L2RoeRiemannSolverDissipation : public RoeBaseRiemannSolverDissipation<dim, nstate, real>
{
public:
    /// Constructor
    explicit L2RoeRiemannSolverDissipation(std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : RoeBaseRiemannSolverDissipation<dim, nstate, real>(physics_input){};

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

/// Base class of numerical flux associated with convection
template<int dim, int nstate, typename real>
class NumericalFluxConvective
{
public:
    /// Constructor
    NumericalFluxConvective(
        std::unique_ptr< BaselineNumericalFluxConvective<dim,nstate,real> > baseline_input,
        std::unique_ptr< RiemannSolverDissipation<dim,nstate,real> > riemann_solver_dissipation_input);

protected:
    /// Baseline convective numerical flux object
    std::unique_ptr< BaselineNumericalFluxConvective<dim,nstate,real> > baseline;

    /// Upwind convective numerical flux object
    std::unique_ptr< RiemannSolverDissipation<dim,nstate,real> > riemann_solver_dissipation;

public:
    /// Returns the convective numerical flux at an interface.
    std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const dealii::Tensor<1,dim,real> &normal1) const;
};

/// Lax-Friedrichs numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class LaxFriedrichs : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit LaxFriedrichs(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);

};

/// Roe-Pike numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class RoePike : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit RoePike(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// L2Roe numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class L2Roe : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit L2Roe(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// Central numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class Central : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit Central(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// Entropy conserving numerical flux. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class EntropyConserving : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit EntropyConserving(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// Entropy conserving numerical flux with Lax Friedrichs dissipation. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class EntropyConservingWithLaxFriedrichsDissipation : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit EntropyConservingWithLaxFriedrichsDissipation(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// Entropy conserving numerical flux with Roe dissipation. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class EntropyConservingWithRoeDissipation : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit EntropyConservingWithRoeDissipation(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

/// Entropy conserving numerical flux with L2Roe dissipation. Derived from NumericalFluxConvective.
template<int dim, int nstate, typename real>
class EntropyConservingWithL2RoeDissipation : public NumericalFluxConvective<dim, nstate, real>
{
public:
    /// Constructor
    explicit EntropyConservingWithL2RoeDissipation(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
};

} /// NumericalFlux namespace
} /// PHiLiP namespace

#endif
