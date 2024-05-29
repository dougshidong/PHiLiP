#ifndef __INITIAL_CONDITION_FUNCTION_H__
#define __INITIAL_CONDITION_FUNCTION_H__

// for the initial condition function:
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"
#include "../euler.h" // for FreeStreamInitialConditions
#include "../navier_stokes.h" // for InitialConditionFunction_TurbulentChannelFlow

namespace PHiLiP {

/// Initial condition function used to initialize a particular flow setup/case
template <int dim, int nstate, typename real>
class InitialConditionFunction : public dealii::Function<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor
    InitialConditionFunction();

    /// Value of the initial condition
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

};

/// Function used to evaluate farfield conservative solution
template <int dim, int nstate, typename real>
class FreeStreamInitialConditions : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Farfield conservative solution
    std::array<double,nstate> farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store farfield_conservative solution
     */
    explicit FreeStreamInitialConditions (const Physics::Euler<dim,nstate,double> euler_physics)
            : InitialConditionFunction<dim,nstate,real>()
    {
        //const double density_bc = 2.33333*euler_physics.density_inf;
        const double density_bc = euler_physics.density_inf;
        const double pressure_bc = 1.0/(euler_physics.gam*euler_physics.mach_inf_sqr);
        std::array<double,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = density_bc;
        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = euler_physics.velocities_inf[d]; }
        primitive_boundary_values[nstate-1] = pressure_bc;
        farfield_conservative = euler_physics.convert_primitive_to_conservative(primitive_boundary_values);
    }

    /// Returns the istate-th farfield conservative value
    double value (const dealii::Point<dim> &/*point*/, const unsigned int istate) const
    {
        return farfield_conservative[istate];
    }
};

/// Function used to evaluate initial turbulent channel conservative solution
template <int dim, int nstate, typename real>
class InitialConditionFunction_TurbulentChannelFlow : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor.
    InitialConditionFunction_TurbulentChannelFlow (
        const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
        const double channel_friction_velocity_reynolds_number_,
        const double domain_length_x_,
        const double domain_length_y_,
        const double domain_length_z_);

    const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics; ///< Navier-Stokes physics object
    const double channel_friction_velocity_reynolds_number; ///< Channel Reynolds number based on wall friction velocity
    const double domain_length_x; ///< Domain length in x-direction
    const double domain_length_y; ///< Domain length in y-direction
    const double domain_length_z; ///< Domain length in z-direction
    const double channel_height; ///< Channel height
    const double half_channel_height; ///< Half channel height

    /// Value of initial condition expressed in terms of conservative variables
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

protected:
    /// distance from closest wall
    real get_distance_from_wall(const dealii::Point<dim,real> &point) const;
    /// x-velocity
    virtual real x_velocity (const dealii::Point<dim,real> &point, const real density, const real temperature) const;
    /// y-velocity
    real y_velocity (const dealii::Point<dim,real> &point) const;
};

template <int dim, int nstate, typename real>
class InitialConditionFunction_TurbulentChannelFlow_Turbulent : public InitialConditionFunction_TurbulentChannelFlow<dim,nstate,real>
{
public:
    /// Constructor.
    /** Initial condition based on the Reichardt function
     *  Reference: https://how4.cenaero.be/sites/how4.cenaero.be/files/BS2_ChannelFlowRe590.pdf
     */
    InitialConditionFunction_TurbulentChannelFlow_Turbulent (
        const Physics::NavierStokes<dim,nstate,double> navier_stokes_physics_,
        const double channel_friction_velocity_reynolds_number_,
        const double domain_length_x_,
        const double domain_length_y_,
        const double domain_length_z_);

protected:
    /// x-velocity
    real x_velocity (const dealii::Point<dim,real> &point, const real density, const real temperature) const override;
};

/// Initial Condition Function: Euler Equations (primitive values)
template <int dim, int nstate, typename real>
class InitialConditionFunction_EulerBase : public InitialConditionFunction<dim, nstate, real>
{
protected:
    using dealii::Function<dim, real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for test cases using Euler equations.
    explicit InitialConditionFunction_EulerBase(
        Parameters::AllParameters const* const param);

    /// Value of initial condition expressed in terms of conservative variables
    real value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const override;

protected:
    /// Value of initial condition expressed in terms of primitive variables
    virtual real primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const = 0;

    /// Converts value from: primitive to conservative
    real convert_primitive_to_conversative_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const;

private:
    // Euler physics pointer. Used to convert primitive to conservative.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;
};

/// Initial Condition Function: Taylor Green Vortex (uniform density)
template <int dim, int nstate, typename real>
class InitialConditionFunction_TaylorGreenVortex : public InitialConditionFunction_EulerBase<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for TaylorGreenVortex_InitialCondition with uniform density
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  Reference: (1) Gassner2016split, 
     *             (2) de la Llave Plata et al. (2019). "On the performance of a high-order multiscale DG approach to LES at increasing Reynolds number."
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    explicit InitialConditionFunction_TaylorGreenVortex (
            Parameters::AllParameters const *const param);

    const double gamma_gas; ///< Constant heat capacity ratio of fluid.
    const double mach_inf; ///< Farfield Mach number.
    const double mach_inf_sqr; ///< Farfield Mach number squared.

protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Value of initial condition for density
    virtual real density(const dealii::Point<dim,real> &point) const;
};

/// Initial Condition Function: Taylor Green Vortex (isothermal density)
template <int dim, int nstate, typename real>
class InitialConditionFunction_TaylorGreenVortex_Isothermal
    : public InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
{
public:
    /// Constructor for TaylorGreenVortex_InitialCondition with isothermal density
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  -- Reference: (1) Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                    between the spectral difference and flux reconstruction schemes." 
     *                    Computers & Fluids 221 (2021): 104922.
     *                (2) Brian Vermeire 2014 Thesis  
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    explicit InitialConditionFunction_TaylorGreenVortex_Isothermal (
            Parameters::AllParameters const *const param);

protected:
    /// Value of initial condition for density
    real density(const dealii::Point<dim,real> &point) const override;
};

/// Initial Condition Function: 1D Burgers Rewienski
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersRewienski: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersRewienski ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Viscous
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersViscous: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersViscous ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersInviscid: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersInviscid
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersInviscid ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersInviscidEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersInviscidEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersInviscidEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid
template <int dim, int nstate, typename real>
class InitialConditionFunction_Advection
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_Inviscid
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_Advection ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Advection Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_AdvectionEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_AdvectionEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_AdvectionEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Convection Diffusion Orders of Accuracy
template <int dim, int nstate, typename real>
class InitialConditionFunction_ConvDiff
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_ConvDiffEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_ConvDiff ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Convection Diffusion Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_ConvDiffEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_ConvDiffEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_ConvDiffEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: 1D Sine Function; used for temporal convergence
template <int dim, int nstate, typename real>
class InitialConditionFunction_1DSine
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_1DSine
    InitialConditionFunction_1DSine ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: Isentropic vortex
template <int dim, int nstate, typename real>
class InitialConditionFunction_IsentropicVortex
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_IsentropicVortex
    /** Setup according to "A Survey of the Isentropic Euler Vortex Problem using High-Order Methods"
     *     Spiegel et al., 2015
     *  Using "Shu" variant (first row of Table 1)
     *  Non-dimensional initialization, i.e. directly using Table 1
     *  Increased domain from L=5 -> L=10 per recommendation of Spiegel et al
     */
    explicit InitialConditionFunction_IsentropicVortex (
            Parameters::AllParameters const *const param);

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

protected:

    // Euler physics pointer. Used to convert primitive to conservative.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

};



/// Kelvin-Helmholtz Instability, parametrized by Atwood number
/** See Chan et al., On the entropy projection..., 2022, Pg. 15
 *      Note that some equations are not typed correctly
 *      See github.com/trixi-framework/paper-2022-robustness-entropy-projection
 *      for initial condition which is implemented herein
 */
template <int dim, int nstate, typename real>
class InitialConditionFunction_KHI : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on
    
public:
    /// Constructor
    explicit InitialConditionFunction_KHI(
            Parameters::AllParameters const *const param);

    /// Value of initial condition
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

protected:
    
    /// Atwood number: quantifies density difference.
    const real atwood_number;

    // Euler physics pointer. Used to convert primitive to conservative.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

};

/// Initial Condition Function: 1D Sod Shock Tube
/** See Chen & Shu, Entropy stable high order discontinuous 
*   Galerkin methods with suitable quadrature rules for hyperbolic 
*   conservation laws., 2017, Pg. 25
*/
template <int dim, int nstate, typename real>
class InitialConditionFunction_SodShockTube: public InitialConditionFunction_EulerBase<dim,nstate,real>
{
protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const override;

public:
    /// Constructor for InitialConditionFunction_SodShockTube
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit InitialConditionFunction_SodShockTube (
            Parameters::AllParameters const* const param);
};

/// Initial Condition Function: 2D Low Density Euler
/** See Zhang & Shu, On positivity-preserving high order 
*   discontinuous Galerkin schemes for compressible Euler 
*   equations on rectangular meshes, 2010 Pg. 10
*/
template <int dim, int nstate, typename real>
class InitialConditionFunction_LowDensity2D: public InitialConditionFunction_EulerBase<dim,nstate,real>
{
protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const override;

public:
    /// Constructor for InitialConditionFunction_SodShockTube
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit InitialConditionFunction_LowDensity2D(
        Parameters::AllParameters const* const param);
};

/// Initial Condition Function: 1D Leblanc Shock Tube
/** See Zhang & Shu, On positivity-preserving high order
*   discontinuous Galerkin schemes for compressible Euler
*   equations on rectangular meshes, 2010 Pg. 14
*/
template <int dim, int nstate, typename real>
class InitialConditionFunction_LeblancShockTube : public InitialConditionFunction_EulerBase<dim, nstate, real>
{
protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const override;

public:
    /// Constructor for InitialConditionFunction_SodShockTube
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit InitialConditionFunction_LeblancShockTube(
        Parameters::AllParameters const* const param);
};

/// Initial Condition Function: 1D Shu Osher Problem
/** See Johnsen et al., Assessment of high-resolution methods 
*   for numerical simulations of compressible turbulence with 
*   shock waves, 2010 Pg. 7
*/
template <int dim, int nstate, typename real>
class InitialConditionFunction_ShuOsherProblem : public InitialConditionFunction_EulerBase<dim, nstate, real>
{
protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim, real>& point, const unsigned int istate = 0) const override;

public:
    /// Constructor for InitialConditionFunction_SodShockTube
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit InitialConditionFunction_ShuOsherProblem(
        Parameters::AllParameters const* const param);
};

/// Initial condition 0.
template <int dim, int nstate, typename real>
class InitialConditionFunction_Zero : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on
    
public:
    /// Constructor
    InitialConditionFunction_Zero();

    /// Returns zero.
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial condition function factory
template <int dim, int nstate, typename real>
class InitialConditionFactory
{
protected:    
    /// Enumeration of all flow solver initial conditions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    /// Enumeration of all taylor green vortex initial condition sub-types defined in the Parameters class
    using DensityInitialConditionEnum = Parameters::FlowSolverParam::DensityInitialConditionType;
    /// Enumeration of all turbulent channel flow inition condition sub-types defined in the Parameters class
    using XVelocityInitialConditionEnum = Parameters::FlowSolverParam::XVelocityInitialConditionType;

public:
    /// Construct InitialConditionFunction object from global parameter file
    static std::shared_ptr<InitialConditionFunction<dim,nstate,real>>
    create_InitialConditionFunction(
        Parameters::AllParameters const *const param);
};

} // PHiLiP namespace
#endif
