#ifndef __FE_VALUES_SHAPE_HESSIAN_H__
#define __FE_VALUES_SHAPE_HESSIAN_H__

#include <deal.II/fe/fe_values.h>

namespace PHiLiP {
/// Class to evaluate hessians of shape functions w.r.t. physical quadrature points.
template<int dim>
class FEValuesShapeHessian {

public:
    /// Constructor
    FEValuesShapeHessian() {};

    /// Destructor
    ~FEValuesShapeHessian() {};
    
    /// Store inverse jacobian and 3rd order tensors which are common at a quadrature point.
    void reinit();

    /// Evaluates hessian of shape functions w.r.t. phyical quadrature points.
    dealii::Tensor<2,dim,double> shape_hessian_component();

private:
    /// Stores inverse jacobian of mapping betwenn reference and physical cell.
    dealii::Tensor<2,dim,double> inverse_jacobian;
    
    /// Stores transpose of inverse jacobian of mapping betwenn reference and physical cell.
    dealii::Tensor<2,dim,double> inverse_jacobian_transpose;

    /// Stores derivative of the jacobian inverse w.r.t. physical quadrature point.
    dealii::Tensor<3,dim,double> derivative_jacobian_inverse_wrt_phys_q; 
};
} // PHiLiP namespace
#endif
