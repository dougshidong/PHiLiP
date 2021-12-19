#ifndef __MANUFACTUREDSOLUTIONFUNCTION_H__
#define __MANUFACTUREDSOLUTIONFUNCTION_H__

#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>

//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
#include "parameters/all_parameters.h"


namespace PHiLiP {


/// Manufactured solution used for grid studies to check convergence orders.
/** This class also provides derivatives necessary  to evaluate source terms.
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
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

    /// Gradient of the exact manufactured solution
    /** \code
     *  grad_u[s][0] = A[s]*freq[s][0]*cos(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  grad_u[s][1] = A[s]*freq[s][1]*sin(freq[s][0]*x)*cos(freq[s][1]*y)*sin(freq[s][2]*z);
     *  grad_u[s][2] = A[s]*freq[s][2]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*cos(freq[s][2]*z);
     *  \endcode
     */
    virtual dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

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
    virtual dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;

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

protected:
    ///@{
    /** Constants used to manufactured solution.
     */
    std::vector<double> base_values;
    std::vector<double> amplitudes;
    std::vector<dealii::Tensor<1,dim,real>> frequencies;
    //@}
};

/// Product of sine waves manufactured solution
template <int dim, typename real>
class ManufacturedSolutionSine 
    : public ManufacturedSolutionFunction<dim, real>
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

public:
    /// Constructor
    ManufacturedSolutionSine(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Product of cosine waves manufactured solution
template <int dim, typename real>
class ManufacturedSolutionCosine 
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionCosine(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Sum of sine waves manufactured solution
template <int dim, typename real>
class ManufacturedSolutionAdd 
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionAdd(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Sum of exponential functions manufactured solution
template <int dim, typename real>
class ManufacturedSolutionExp
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionExp(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Sum of polynomial manufactured solution
template <int dim, typename real>
class ManufacturedSolutionPoly 
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionPoly(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Sum of even order polynomial functions manufactured solution
template <int dim, typename real>
class ManufacturedSolutionEvenPoly 
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionEvenPoly(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Hump manufactured solution based on arctangent functions
template <int dim, typename real>
class ManufacturedSolutionAtan
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionAtan(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate)
    {
        n_shocks.resize(dim);
        S_j.resize(dim);
        x_j.resize(dim);

        for(unsigned int i = 0; i<dim; ++i){
            n_shocks[i] = 2;

            S_j[i].resize(n_shocks[i]);
            x_j[i].resize(n_shocks[i]);

            // S_j[i][0] =  10;
            // S_j[i][1] = -10;

            S_j[i][0] =  50;
            S_j[i][1] = -50;

            x_j[i][0] = -1/sqrt(2);
            x_j[i][1] =  1/sqrt(2);

            // x_j[i][0] = 1-1/sqrt(2);
            // x_j[i][1] = 1/sqrt(2);
        }
    }
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

private:
    std::vector<unsigned int> n_shocks; ///< number of shocks
    std::vector<std::vector<real>> S_j; ///< shock strengths
    std::vector<std::vector<real>> x_j; ///< shock positions
};

/// Scalar boundary layer manufactured solution
template <int dim, typename real>
class ManufacturedSolutionBoundaryLayer
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionBoundaryLayer(const unsigned int nstate = 1)
        : ManufacturedSolutionFunction<dim,real>(nstate)
        , epsilon(nstate)
    {
        for(int istate = 0; istate < (int)nstate; ++istate){
            for (int d=0; d<dim; d++){
                // epsilon[istate][d] = 0.1;   // smooth
                epsilon[istate][d] = 0.005; // strong
            }
        }
    }
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

private:
    /// Boundary layer strength parameter
    std::vector<dealii::Tensor<1,dim,real>> epsilon;
};

/// S-Shock manufactured solution
template <int dim, typename real>
class ManufacturedSolutionSShock 
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionSShock(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate)
    {
        // setting constant for function
        // f(x,y) = a * tanh(b*sin(c*y + d) + e*x + f)

        // Ekelschot
        // Note: form given does not have brackets around b*(...)
        // a =  0.75;
        // b =  2.0;
        // c =  5.0;
        // d =  0.0;
        // e = -6.0;
        // f =  0.0;

        double scale_atan = 2.0;

        // shifted from [-1,1]^2 -> [0,1]
        a =   0.75;
        b =   2.0*scale_atan;
        c =  10.0;
        d =  -5.0;
        e = -12.0*scale_atan;
        f =   6.0*scale_atan;
    }
    /// Value
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

private:
    ///@{
    /// equation constants
    real a, b, c, d, e, f; 
    //@}
};

/// Quadratic function manufactured solution
template <int dim, typename real>
class ManufacturedSolutionQuadratic
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionQuadratic(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate)
    {
        // assigning the scaling coeffs for hessian diagonals
        for(unsigned int d = 0; d < dim; ++d){
            // diag(1, 4, 9, ...)
            alpha_diag[d] = (d+1)*(d+1);
        }
    }
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

private:
    std::array<real, dim> alpha_diag; ///< Diagonal hessian component scaling
};


/// Sum of Alex functions manufactured solution (was used in Manufactured Solution presentation).
template <int dim, typename real>
class ManufacturedSolutionAlex
    : public ManufacturedSolutionFunction<dim, real>
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
public:
    ///Constructor
    ManufacturedSolutionAlex(const unsigned int nstate = 1)
        :   ManufacturedSolutionFunction<dim,real>(nstate){}
    /// Value
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Navah and Nadarajah free flows manufactured solution base
/// Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
template <int dim, typename real>
class ManufacturedSolutionNavahBase
: public ManufacturedSolutionFunction<dim, real>
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
public:
    /// Constructor
    ManufacturedSolutionNavahBase(const unsigned int nstate = 4)
        :   ManufacturedSolutionFunction<dim,real>(nstate)
    {
        // static_assert(dim==2, "ManufacturedSolutionNavahBase() should be created with dim=2");
        // static_assert(nstate==dim+2, "ManufacturedSolutionNavahBase() should be created with nstate=dim+2");
        
        const double pi = atan(1)*4.0;///< pi constant
        real L = 1.0;  ///< reference length
        c = pi/L; ///< constant
    }
    /// Value of conservative variables
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Gradient of conservative variables
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
    /// Hessian of conservative variables
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
protected:
    std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm); placeholder
    real c; ///< Constant, pi/L
    /// Value of primitive variables
    real primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Gradient of primitive variables
    dealii::Tensor<1,dim,real> primitive_gradient (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    /// Hessian of primitive variables
    dealii::SymmetricTensor<2,dim,real> primitive_hessian (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
};

/// Navah and Nadarajah free flows manufactured solution: MS1
template <int dim, typename real>
class ManufacturedSolutionNavah_MS1
    : public ManufacturedSolutionNavahBase<dim, real>
{
public:
    /** Constructor for MS-1
     *  Sets the Navah Coefficient Matrix for the specified navah_solution.
     *  Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
     *  Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
     */
    ManufacturedSolutionNavah_MS1(const unsigned int nstate = 4)
        :   ManufacturedSolutionNavahBase<dim,real>(nstate)
    {
        std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm)
        /* MS-1 */
        ncm[0][0]= 1.0; ncm[0][1]=0.3; ncm[0][2]=-0.2; ncm[0][3]=0.3; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 1.0; ncm[1][1]=0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 1.0; ncm[2][1]=0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=18.0; ncm[3][1]=5.0; ncm[3][2]= 5.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        }
        this->ncm=ncm; // done this way to minimize the use of keyword "this->"
    }
};

/// Navah and Nadarajah free flows manufactured solution: MS2
template <int dim, typename real>
class ManufacturedSolutionNavah_MS2
    : public ManufacturedSolutionNavahBase<dim, real>
{
public:
    /** Constructor for MS-2
     *  Sets the Navah Coefficient Matrix for the specified navah_solution.
     *  Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
     *  Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
     */
    ManufacturedSolutionNavah_MS2(const unsigned int nstate = 4)
        :   ManufacturedSolutionNavahBase<dim,real>(nstate)
    {
        std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm)
        /* MS-2 */
        ncm[0][0]=2.7; ncm[0][1]=0.9; ncm[0][2]=-0.9; ncm[0][3]=1.0; ncm[0][4]=1.5; ncm[0][5]=1.5; ncm[0][6]=1.5;
        ncm[1][0]=2.0; ncm[1][1]=0.7; ncm[1][2]= 0.7; ncm[1][3]=0.4; ncm[1][4]=1.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]=2.0; ncm[2][1]=0.4; ncm[2][2]= 0.4; ncm[2][3]=0.4; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=2.0; ncm[3][1]=1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=1.0; ncm[3][5]=1.0; ncm[3][6]=1.5;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        }
        this->ncm=ncm; // done this way to minimize the use of keyword "this->"
    }
};

/// Navah and Nadarajah free flows manufactured solution: MS3
template <int dim, typename real>
class ManufacturedSolutionNavah_MS3
    : public ManufacturedSolutionNavahBase<dim, real>
{
public:
    /** Constructor for MS-3
     *  Sets the Navah Coefficient Matrix for the specified navah_solution.
     *  Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
     *  Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
     */
    ManufacturedSolutionNavah_MS3(const unsigned int nstate = 4)
        :   ManufacturedSolutionNavahBase<dim,real>(nstate)
    {
        std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm)
        /* MS-3 */
        ncm[0][0]= 1.0; ncm[0][1]=0.1; ncm[0][2]=-0.2; ncm[0][3]=0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]=0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]=0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]=1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        }
        this->ncm=ncm; // done this way to minimize the use of keyword "this->"
    }
};

/// Navah and Nadarajah free flows manufactured solution: MS4
template <int dim, typename real>
class ManufacturedSolutionNavah_MS4
    : public ManufacturedSolutionNavahBase<dim, real>
{
public:
    /** Constructor for MS-4
     *  Sets the Navah Coefficient Matrix for the specified navah_solution.
     *  Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
     *  Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
     */
    ManufacturedSolutionNavah_MS4(const unsigned int nstate = 4)
        :   ManufacturedSolutionNavahBase<dim,real>(nstate)
    {
        std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm)
        /* MS-4 */
        ncm[0][0]= 1.0; ncm[0][1]=  0.1; ncm[0][2]= -0.2; ncm[0][3]= 0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]=  0.3; ncm[1][2]=  0.3; ncm[1][3]= 0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]=  0.3; ncm[2][2]=  0.3; ncm[2][3]= 0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]=  1.0; ncm[3][2]=  1.0; ncm[3][3]= 0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        ncm[4][0]= 0.6; ncm[4][1]=-0.03; ncm[4][2]=-0.02; ncm[4][3]=0.02; ncm[4][4]=2.0; ncm[4][5]=1.0; ncm[4][6]=3.0;
        this->ncm=ncm; // done this way to minimize the use of keyword "this->"
    }
};

/// Navah and Nadarajah free flows manufactured solution: MS5
template <int dim, typename real>
class ManufacturedSolutionNavah_MS5
    : public ManufacturedSolutionNavahBase<dim, real>
{
public:
    /** Constructor for MS-5
     *  Sets the Navah Coefficient Matrix for the specified navah_solution.
     *  Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
     *  Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
     */
    ManufacturedSolutionNavah_MS5(const unsigned int nstate = 4)
        :   ManufacturedSolutionNavahBase<dim,real>(nstate)
    {
        std::array<dealii::Tensor<1,7,double>,5> ncm; ///< Navah Coefficient Matrix (ncm)
        /* MS-5 */
        ncm[0][0]= 1.0; ncm[0][1]= 0.1; ncm[0][2]=-0.2; ncm[0][3]=0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]= 0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]= 0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]= 1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        ncm[4][0]=-6.0; ncm[4][1]=-0.3; ncm[4][2]=-0.2; ncm[4][3]=0.2; ncm[4][4]=2.0; ncm[4][5]=1.0; ncm[4][6]=3.0;
        this->ncm=ncm; // done this way to minimize the use of keyword "this->"
    }
};


/// Manufactured solution function factory
/** Based on input from Parameters file, generates a standard form
  * of manufactured solution function with suitable value, gradient 
  * and hessian functions for the chosen distribution type.
  * 
  * Functions are selected from enumerator list in 
  * Parameters::ManufacturedSolutionParam::ManufacturedSolutionType
  * 
  * Some Manufactured solutions included additional scaling constants
  * that can also be can also be controlled from the parameter file
  */ 
template <int dim, typename real>
class ManufacturedSolutionFactory
{
    /// Enumeration of all manufactured solution types defined in the Parameters class
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
public:
    /// Construct Manufactured solution object from global parameter file
    static std::shared_ptr< ManufacturedSolutionFunction<dim,real> > 
    create_ManufacturedSolution(
        Parameters::AllParameters const *const param, 
        int                                    nstate);

    /// Construct Manufactured solution object from enumerator list
    static std::shared_ptr< ManufacturedSolutionFunction<dim,real> >
    create_ManufacturedSolution(
        ManufacturedSolutionEnum solution_type,
        int                      nstate);

};

} // namespace PHiLiP

#endif //__MANUFACTUREDSOLUTIONFUNCTION_H__
