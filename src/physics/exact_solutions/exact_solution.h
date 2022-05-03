#ifndef __EXACT_SOLUTION_H__
#define __EXACT_SOLUTION_H__

// for the exact_solution function
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"

namespace PHiLiP {

/// Initial condition function used to initialize a particular flow setup/case
template <int dim, int nstate, typename real>
class ExactSolutionFunction : public dealii::Function<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor
    ExactSolutionFunction();
    /// Destructor
    ~ExactSolutionFunction() {};

    /// Value of the initial condition
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;
};

/// Initial Condition Function: Zero Function; used as a placeholder when there is no exact solution
template <int dim, int nstate, typename real>
class ExactSolutionFunction_Zero
        : public ExactSolutionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    ExactSolutionFunction_Zero ();

    /// Time at which to compute the exact solution
    double t; 

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Sine Function; used for temporal convergence
template <int dim, int nstate, typename real>
class ExactSolutionFunction_1DSine
        : public ExactSolutionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    ExactSolutionFunction_1DSine (double time_compare);

    /// Time at which to compute the exact solution
    double t; 

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};


/// Initial condition function factory
template <int dim, int nstate, typename real>
class ExactSolutionFactory
{
protected:    
    /// Enumeration of all flow solver initial conditions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;

public:
    /// Construct InitialConditionFunction object from global parameter file
    static std::shared_ptr<ExactSolutionFunction<dim,nstate,real>>
    create_ExactSolutionFunction(
        Parameters::AllParameters const *const param);
};

} // PHiLiP namespace
#endif
