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
    for(unsigned int i_phys=0; i_phys<dim; ++i_phys)
    {
        for(unsigned int j_ref=0; j_ref<dim; ++j_ref)
        {
            for(unsigned int k_phys=0; k_phys<dim; ++k_phys)
            {
                for(unsigned int n_phys=0; n_phys<dim; ++n_phys)
                {
                    derivative_jacobian_inverse_wrt_phys_q[j_ref][i_phys][k_phys] -= jacobian_pushed_forward_grad[n_phys][i_phys][k_phys]*jacobian_inverse[j_ref][n_phys];
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
    for(unsigned int i_phys=0; i_phys<dim; ++i_phys)
    {
        for(unsigned int j_phys=0; j_phys<dim; ++j_phys)
        {
            for(unsigned int k_ref=0; k_ref<dim; ++k_ref)
            {
                for(unsigned int l_ref=0; l_ref<dim; ++l_ref)
                {
                    phys_hessian_term1[i_phys][j_phys] += jacobian_inverse[k_ref][i_phys] * shape_hessian_ref[k_ref][l_ref] * jacobian_inverse[l_ref][j_phys];
                }
            }
        }
    }

    // Computing second term: \varphi_{\epsilon}^T * d/dx( J^{-1} )
    // Using v^T*E_xx (i,j)  =  v_k E_xx(k,i,j) = v_k (d^2 E_k/(dx_i dx_j), with E_xx a third order tensor and v a vector.
    dealii::Tensor<2,dim,double> phys_hessian_term2; // initilized to 0
    for(unsigned int i_phys=0; i_phys<dim; ++i_phys)
    {
        for(unsigned int j_phys=0; j_phys<dim; ++j_phys)
        {
            for(unsigned int k_ref=0; k_ref<dim; ++k_ref)
            {
                phys_hessian_term2[i_phys][j_phys] += shape_grad_ref[k_ref] * derivative_jacobian_inverse_wrt_phys_q[k_ref][i_phys][j_phys];
            }
        }
    }

    dealii::Tensor<2,dim,double> shape_hessian_phys = phys_hessian_term1;
    shape_hessian_phys += phys_hessian_term2;
    return shape_hessian_phys;
}

template class FEValuesShapeHessian<PHILIP_DIM>;
} // PHiLiP namespace 
