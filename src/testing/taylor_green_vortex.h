#ifndef __INITIALCONDITIONFUNCTION_H__
#define __INITIALCONDITIONFUNCTION_H__

#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>

//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
// #include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {


/// Manufactured solution used for grid studies to check convergence orders.
/** This class also provides derivatives necessary  to evaluate source terms.
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
    using dealii::Function<dim,real>::hessian;
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
    virtual dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

    /// See dealii::Function<dim,real>::vector_gradient [TO DO: Is this necessary?]
    // void vector_gradient (const dealii::Point<dim,real> &p,
    //                       std::vector<dealii::Tensor<1,dim, real> > &gradients) const;
protected:

};

template <int dim, typename real>
class TaylorGreenVortex_InitialCondition
    : public InitialConditionFunction_FlowSolver<dim,real>
{
// We want the Point to be templated on the type,
// however, dealii does not template that part of the Function.
// Therefore, we end up overloading the functions and need to "import"
// those non-overloaded functions to avoid the warning -Woverloaded-virtual
// See: https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
    using dealii::Function<dim,real>::vector_gradient;
public:
    /// Constructor for TaylorGreenVortex_InitialCondition
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  Reference: Gassner2016split, plata2019performance
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    TaylorGreenVortex_InitialCondition (
        const unsigned int nstate,
        const double       gamma_gas,
        const double       mach_inf_sqr);

    const double gamma_gas; ///< Constant heat capacity ratio of fluid.
    const double mach_inf_sqr; ///< Farfield Mach number squared.
        
    /// Value of initial condition expressed in terms of conservative variables
    real value (const dealii::Point<dim> &point, const unsigned int istate = 0) const override;
    /// Gradient of initial condition expressed in terms of conservative variables
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Gradient of initial condition expressed in terms of primitive variables
    dealii::Tensor<1,dim,real> primitive_gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Hessian of initial condition expressed in terms of primitive variables
    dealii::SymmetricTensor<2,dim,real> primitive_hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Converts value from: primitive to conservative
    real convert_primitive_to_conversative_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Converts gradient from: primitive to conservative
    dealii::Tensor<1,dim,real> convert_primitive_to_conversative_gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
};

// protected:
//     using dealii::Function<dim,real>::value;

} // Tests namespace
} // PHiLiP namespace
#endif