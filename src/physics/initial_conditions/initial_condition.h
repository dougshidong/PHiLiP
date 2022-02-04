#ifndef __INITIAL_CONDITION_H__
#define __INITIAL_CONDITION_H__

// for the initial condition function:
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"
#include "../euler.h" // for FreeStreamInitialConditions

namespace PHiLiP {

namespace Physics {
/// Function used to evaluate farfield conservative solution
template <int dim, int nstate>
class FreeStreamInitialConditions : public dealii::Function<dim>
{
public:
    /// Farfield conservative solution
    std::array<double,nstate> farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store farfield_conservative solution
     */
    FreeStreamInitialConditions (const Physics::Euler<dim,nstate,double> euler_physics)
    : dealii::Function<dim,double>(nstate)
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
} // Physics namespace

/// Initial condition function used to initialize a particular flow setup/case
template <int dim, typename real>
class InitialConditionFunction : public dealii::Function<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    const unsigned int nstate; ///< Corresponds to n_components in the dealii::Function
    /// Constructor
    InitialConditionFunction(const unsigned int nstate = 5);
    /// Destructor
    ~InitialConditionFunction() {};

    /// Value of the initial condition
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;
};

/// Initial Condition Function: Taylor Green Vortex
template <int dim, typename real>
class InitialConditionFunction_TaylorGreenVortex
    : public InitialConditionFunction<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for TaylorGreenVortex_InitialCondition
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  Reference: Gassner2016split, plata2019performance
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    InitialConditionFunction_TaylorGreenVortex (
        const unsigned int nstate = 5,
        const double       gamma_gas = 1.4,
        const double       mach_inf = 0.1);

    const double gamma_gas; ///< Constant heat capacity ratio of fluid.
    const double mach_inf; ///< Farfield Mach number.
    const double mach_inf_sqr; ///< Farfield Mach number squared.
        
    /// Value of initial condition expressed in terms of conservative variables
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    
    /// Converts value from: primitive to conservative
    real convert_primitive_to_conversative_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
};

/// Initial Condition Function: 1D Burgers Rewienski
template <int dim, typename real>
class InitialConditionFunction_BurgersRewienski
        : public InitialConditionFunction<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersRewienski (const unsigned int nstate = 1);

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial condition function factory
template <int dim, typename real>
class InitialConditionFactory
{
protected:    
    /// Enumeration of all flow solver initial conditions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;

public:
    /// Construct InitialConditionFunction object from global parameter file
    static std::shared_ptr< InitialConditionFunction<dim,real> > 
    create_InitialConditionFunction(
        Parameters::AllParameters const *const param, 
        int                                    nstate);
};

/// Initial condition 0.
template <int dim, typename real>
class InitialConditionFunction_Zero : public dealii::Function<dim>
{
public:
    /// Constructor to initialize dealii::Function
    InitialConditionFunction_Zero(const unsigned int nstate) 
    : dealii::Function<dim,real>(nstate)
    { }

    /// Returns zero.
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

} // PHiLiP namespace
#endif
