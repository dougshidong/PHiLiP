#include "fe_values_shape_hessian.h"

namespace PHiLiP {

template<int dim>
dealii::Tensor<2,dim,double> FEValuesShapeHessian<dim> :: shape_hessian_component()
{
    dealii::Tensor<2,dim,double> zero_tensor;
    return zero_tensor;
}

template class FEValuesShapeHessian<PHILIP_DIM>;
} // PHiLiP namespace 
