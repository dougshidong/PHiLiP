#ifndef __FE_VALUES_SHAPE_HESSIAN_H__
#define __FE_VALUES_SHAPE_HESSIAN_H__

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

namespace PHiLiP {
/// Class to evaluate hessians of shape functions w.r.t. physical quadrature points.
/** This class is designed for computing shape hessians w.r.t. the physical domain defined by MappingFEField (with volume nodes vector in HighOrderGrid). 
 */
template<int dim>
class FEValuesShapeHessian {

public:
    /// Constructor
    FEValuesShapeHessian() {};

    /// Destructor
    ~FEValuesShapeHessian() {};
    
    /// Store inverse jacobian and 3rd order tensors which will be the same for a combination of cell/physical quadrature point.
    /** @note Currently, this class is designed to compute hessians at only one quadrature point, as required for computing hessians to get optimal metric field.
     */
    void reinit(const dealii::FEValues<dim,dim> &fe_values_volume, const unsigned int iquad); 

    /// Evaluates hessian of shape functions w.r.t. phyical quadrature points.
    dealii::Tensor<2,dim,double> shape_hessian_component(
        const unsigned int idof, 
        const unsigned int iquad, 
        const unsigned int istate, 
        const dealii::FESystem<dim,dim> &fe_ref) const;

private:
    /// Stores inverse jacobian of mapping between reference and physical cell.
    dealii::Tensor<2,dim,double> jacobian_inverse;
    
    /// Derivative of the jacobian inverse w.r.t. physical quadrature point.
    dealii::Tensor<3,dim,double> derivative_jacobian_inverse_wrt_phys_q;
    
    /// Stores quadrature point in the reference domain.
    dealii::Point<dim> ref_point;
};
} // PHiLiP namespace
#endif
