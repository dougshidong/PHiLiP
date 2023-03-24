#include "fe_values_shape_hessian.h"

namespace PHiLiP {

template<int dim>
void FEValuesShapeHessian<dim> ::  reinit(const dealii::FEValues<dim,dim> &fe_values_volume, const unsigned int iquad)
{
    jacobian_inverse = dealii::Tensor<2,dim,double>(fe_values_volume.inverse_jacobian(iquad));
    const dealii::Tensor<3,dim,double> jacobian_pushed_forward_grad = fe_values_volume.jacobian_pushed_forward_grad(iquad);
    
    ref_point = fe_values_volume.get_quadrature().point(iquad);

    // Compute derivative of jacobian inverse w.r.t. physical coordinates.
    // Using equation from dealii's documentation on Mapping, https://www.dealii.org/current/doxygen/deal.II/classMapping.html.
    derivative_jacobian_inverse_wrt_phys_q = 0;
    for(unsigned int i=0; i<dim; ++i)
    {
        for(unsigned int j=0; j<dim; ++j)
        {
            for(unsigned int k=0; k<dim; ++k)
            {
                for(unsigned int n=0; n<dim; ++n)
                {
                    derivative_jacobian_inverse_wrt_phys_q[j][i][k] -= jacobian_pushed_forward_grad[n][i][k]*jacobian_inverse[j][n];
                }
            }
        }
    }

}

// Had to code this up because shape_hessian_component() hasn't been implemented yet by dealii's MappingFEField.
// This class can be deprecated in future once dealii's shape hessian with MappingFEField works.
template<int dim>
dealii::Tensor<2,dim,double> FEValuesShapeHessian<dim> :: shape_hessian_component(
        const unsigned int idof, 
        const unsigned int /*iquad*/, 
        const unsigned int istate, 
        const dealii::FESystem<dim,dim> &fe_ref) const
{
    dealii::Tensor<1,dim,double> shape_grad_ref = fe_ref.shape_grad_component(idof, ref_point, istate); // \varphi_{\epsilon}
    dealii::Tensor<2,dim,double> shape_hessian_ref = fe_ref.shape_grad_grad_component(idof, ref_point, istate); // \varphi_{\epsilon \epsilon}
    
    // Shape hessian w.r.t. physical x = \varphi_{xx} = J^{-T} \varphi_{\epsilon \epsilon} J^{-1} + \varphi_{\epsilon}^T * d/dx( J^{-1} );

    // Computing first term: J^{-T} \varphi_{\epsilon \epsilon} J^{-1}.
    // Using (A^T*B*A)_{i,j} = a_{ki} * b_{kl} * a_{lj}
    dealii::Tensor<2,dim,double> phys_hessian_term1; // initialized to 0
    for(unsigned int i=0; i<dim; ++i)
    {
        for(unsigned int j=0; j<dim; ++j)
        {
            for(unsigned int k=0; k<dim; ++k)
            {
                for(unsigned int l=0; l<dim; ++l)
                {
                    phys_hessian_term1[i][j] += jacobian_inverse[k][i] * shape_hessian_ref[k][l] * jacobian_inverse[l][j];
                }
            }
        }
    }

    // Computing second term: \varphi_{\epsilon}^T * d/dx( J^{-1} )
    // Using v^T*E_xx (i,j)  =  v_k E_xx(k,i,j) = v_k (d^2 E_k/(dx_i dx_j), with E_xx a third order tensor and v a vector.
    dealii::Tensor<2,dim,double> phys_hessian_term2; // initilized to 0
    for(unsigned int i=0; i<dim; ++i)
    {
        for(unsigned int j=0; j<dim; ++j)
        {
            for(unsigned int k=0; k<dim; ++k)
            {
                phys_hessian_term2[i][j] += shape_grad_ref[k] * derivative_jacobian_inverse_wrt_phys_q[k][i][j];
            }
        }
    }

    dealii::Tensor<2,dim,double> shape_hessian_phys = phys_hessian_term1;
    shape_hessian_phys += phys_hessian_term2;
    return shape_hessian_phys;
}

template class FEValuesShapeHessian<PHILIP_DIM>;
} // PHiLiP namespace 
