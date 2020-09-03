#ifndef __MANUFACTUREDSOLUTIONFUNCTION_H__
#define __MANUFACTUREDSOLUTIONFUNCTION_H__

#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>

//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
//#include "parameters/all_parameters.h"


namespace PHiLiP {


/// Manufactured solution used for grid studies to check convergence orders.
/** This class also provides derivatives necessary to evaluate source terms.
 */
template <int dim, typename real>
class ManufacturedSolutionFunction : public dealii::Function<dim,real>
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
    const unsigned int nstate; ///< Corresponds to n_components in the dealii::Function
    /// Constructor that initializes base_values, amplitudes, frequencies.
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     */
    ManufacturedSolutionFunction (const unsigned int nstate = 1);

    /// Destructor
    ~ManufacturedSolutionFunction() {};
  
    /// Manufactured solution exact value
    /** \code
     *  u[s] = A[s]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  \endcode
     */
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Gradient of the exact manufactured solution
    /** \code
     *  grad_u[s][0] = A[s]*freq[s][0]*cos(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  grad_u[s][1] = A[s]*freq[s][1]*sin(freq[s][0]*x)*cos(freq[s][1]*y)*sin(freq[s][2]*z);
     *  grad_u[s][2] = A[s]*freq[s][2]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*cos(freq[s][2]*z);
     *  \endcode
     */
    virtual dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Uses finite-difference to evaluate the gradient
    dealii::Tensor<1,dim,real> gradient_fd (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Hessian of the exact manufactured solution
    /** \code
     *  hess_u[s][0][0] = -A[s]*freq[s][0]*freq[s][0]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  hess_u[s][0][1] =  A[s]*freq[s][0]*freq[s][1]*cos(freq[s][0]*x)*cos(freq[s][1]*y)*sin(freq[s][2]*z);
     *  hess_u[s][0][2] =  A[s]*freq[s][0]*freq[s][2]*cos(freq[s][0]*x)*sin(freq[s][1]*y)*cos(freq[s][2]*z);
     *
     *  hess_u[s][1][0] =  A[s]*freq[s][1]*freq[s][0]*cos(freq[s][0]*x)*cos(freq[s][1]*y)*sin(freq[s][2]*z);
     *  hess_u[s][1][1] = -A[s]*freq[s][1]*freq[s][1]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  hess_u[s][1][2] =  A[s]*freq[s][1]*freq[s][2]*sin(freq[s][0]*x)*cos(freq[s][1]*y)*cos(freq[s][2]*z);
     *
     *  hess_u[s][2][0] =  A[s]*freq[s][2]*freq[s][0]*cos(freq[s][0]*x)*sin(freq[s][1]*y)*cos(freq[s][2]*z);
     *  hess_u[s][2][1] =  A[s]*freq[s][2]*freq[s][1]*sin(freq[s][0]*x)*cos(freq[s][1]*y)*cos(freq[s][2]*z);
     *  hess_u[s][2][2] = -A[s]*freq[s][2]*freq[s][2]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  \endcode
     *
     *  Note that this term is symmetric since \f$\frac{\partial u }{\partial x \partial y} = \frac{\partial u }{\partial y \partial x} \f$
     */
    virtual dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Uses finite-difference to evaluate the hessian
    dealii::SymmetricTensor<2,dim,real> hessian_fd (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Same as Function::values() except it returns it into a std::vector format.
    std::vector<real> stdvector_values (const dealii::Point<dim,real> &point) const;

  
    /// See dealii::Function<dim,real>::vector_gradient
    void vector_gradient (const dealii::Point<dim,real> &p,
                          std::vector<dealii::Tensor<1,dim, real> > &gradients) const;

    // Virtual functions inherited from dealii::Function
    //
    // virtual real value (const Point<dim,real> &p,
    //                               const unsigned int  component = 0) const;
  
    // virtual void vector_value (const Point<dim,real> &p,
    //                           Vector<real> &values) const;
  
    // virtual void value_list (const std::vector<Point<dim,real> > &points,
    //                         std::vector<real> &values,
    //                         const unsigned int              component = 0) const;
  
    // virtual void vector_value_list (const std::vector<Point<dim,real> > &points,
    //                                std::vector<Vector<real> > &values) const;
  
    // virtual void vector_values (const std::vector<Point<dim,real> > &points,
    //                            std::vector<std::vector<real> > &values) const;
  
    // virtual Tensor<1,dim, real> gradient (const Point<dim,real> &p,
    //                                                 const unsigned int  component = 0) const;
  
    // virtual void gradient_list (const std::vector<Point<dim,real> > &points,
    //                            std::vector<Tensor<1,dim, real> > &gradients,
    //                            const unsigned int              component = 0) const;
  
    // virtual void vector_gradients (const std::vector<Point<dim,real> > &points,
    //                               std::vector<std::vector<Tensor<1,dim, real> > > &gradients) const;
  
    // virtual void vector_gradient_list (const std::vector<Point<dim,real> > &points,
    //                                   std::vector<std::vector<Tensor<1,dim, real> > > &gradients) const;

private:
    ///@{
    /** Constants used to manufactured solution.
     */
    std::vector<double> base_values;
    std::vector<double> amplitudes;
    std::vector<dealii::Tensor<1,dim,real>> frequencies;
    //@}
};

}

#endif
