#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for the initial condition function
// #include <deal.II/lac/vector.h> // TO DO: is this needed?
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"

//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
// #include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {


/// Initial condition function used to initialize a particular flow setup/case.
/** This class also provides derivatives necessary to initialize the boundary gradients.
 */
template <int dim, typename real>
class InitialConditionFunction_FlowSolver : public dealii::Function<dim,real>
{
// We want the Point to be templated on the type,
// however, dealii does not template that part of the Function.
// Therefore, we end up overloading the functions and need to "import"
// those non-overloaded functions to avoid the warning -Woverloaded-virtual
// See: https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    // using dealii::Function<dim,real>::hessian;
    // using dealii::Function<dim,real>::vector_gradient; // TO DO: is this needed?
public:
    const unsigned int nstate; ///< Corresponds to n_components in the dealii::Function
    /// Constructor
    InitialConditionFunction_FlowSolver(const unsigned int nstate);
    /// Destructor
    ~InitialConditionFunction_FlowSolver() {};

    /// Value of the initial condition
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

    /// Gradient of the initial condition
    virtual dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

    /// Hessian of the initial condition
    // virtual dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

    /// See dealii::Function<dim,real>::vector_gradient [TO DO: Is this necessary?]
    // void vector_gradient (const dealii::Point<dim,real> &p,
    //                       std::vector<dealii::Tensor<1,dim, real> > &gradients) const;
protected:

};

/// Initial condition function factory
/** Based on input from Parameters file, generates a standard form
  * of manufactured solution function with suitable value, gradient 
  * and hessian functions for the chosen distribution type.
  * 
  * Functions are selected from enumerator list in 
  * Parameters::FlowSolverParam::FlowCaseType
  */ 
template <int dim, typename real>
class InitialConditionFactory_FlowSolver
{
    /// Enumeration of all flow solver initial conditions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
public:
    /// Construct InitialConditionFunction_FlowSolver object from global parameter file
    static std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> > 
    create_InitialConditionFunction_FlowSolver(
        Parameters::AllParameters const *const param, 
        int                                    nstate);

    /// Construct InitialConditionFunction_FlowSolver object from enumerator list
    static std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> >
    create_InitialConditionFunction_FlowSolver(
        FlowCaseEnum flow_type,
        int          nstate);
};

} // Tests namespace
} // PHiLiP namespace
#endif