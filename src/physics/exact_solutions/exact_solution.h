#ifndef __EXACT_SOLUTION_H__
#define __EXACT_SOLUTION_H__

// for the exact_solution function
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"

namespace PHiLiP {

/// Exact solution function used for a particular flow setup/case
template <int dim, int nstate, typename real>
class ExactSolutionFunction : public dealii::Function<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor
    ExactSolutionFunction();

    /// Value of the exact solution at a point 
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;
};

/// Exact Solution Function: Zero Function; used as a placeholder when there is no exact solution
template <int dim, int nstate, typename real>
class ExactSolutionFunction_Zero
        : public ExactSolutionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for ExactSolutionFunction_Zero
    explicit ExactSolutionFunction_Zero (double time_compare);

    /// Time at which to compute the exact solution
    const double t; 

    /// Value of the exact solution at a point 
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Exact Solution Function: 1D Sine Function; used for temporal convergence
template <int dim, int nstate, typename real>
class ExactSolutionFunction_1DSine
        : public ExactSolutionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for ExactSolutionFunction_1DSine
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit ExactSolutionFunction_1DSine (double time_compare);

    /// Time at which to compute the exact solution
    const double t; 

    /// Value of the exact solution at a point 
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Exact Solution Function: Isentropic vortex 
template <int dim, int nstate, typename real>
class ExactSolutionFunction_IsentropicVortex
        : public ExactSolutionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for ExactSolutionFunction_IsentropicVortex
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    explicit ExactSolutionFunction_IsentropicVortex (double time_compare);

    /// Time at which to compute the exact solution
    const double t; 

    /// Value of the exact solution at a point 
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Exact solution function factory
template <int dim, int nstate, typename real>
class ExactSolutionFactory
{
protected:    
    /// Enumeration of all flow solver exact solutions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;

public:
    /// Construct ExactSolutionFunction object from global parameter file
    static std::shared_ptr<ExactSolutionFunction<dim,nstate,real>>
        create_ExactSolutionFunction(const Parameters::FlowSolverParam& flow_solver_parameters, const double time_compare);
};

} // PHiLiP namespace
#endif
