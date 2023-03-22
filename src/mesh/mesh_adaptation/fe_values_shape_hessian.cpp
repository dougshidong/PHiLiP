#include "fe_values_shape_hessian.h"

namespace PHiLiP {

template<int dim>
void FEValuesShapeHessian<dim> ::  reinit(const dealii::FEValues<dim,dim> &fe_values_volume, const unsigned int iquad)
{
    jacobian_inverse = dealii::Tensor<2,dim,double>(fe_values_volume.inverse_jacobian(iquad));
    jacobian_inverse_transpose = dealii::transpose(jacobian_inverse);
    const dealii::Tensor<2,dim,double> jacobian = dealii::Tensor<2,dim,double>(fe_values_volume.jacobian(iquad));
    const dealii::Tensor<3,dim,double> jacobian_pushed_forward_grad = fe_values_volume.jacobian_pushed_forward_grad(iquad);
    std::cout<<"Jacobian = "<<jacobian<<std::endl;
    std::cout<<"Inverse jacobian = "<<jacobian_inverse<<std::endl;
    std::cout<<"Inverse jacobian transpose = "<<jacobian_inverse_transpose<<std::endl;
    std::cout<<"Jacobian pushed forward grad = "<<jacobian_pushed_forward_grad<<std::endl;
    
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
                    derivative_jacobian_inverse_wrt_phys_q[i][j][k] -= jacobian_pushed_forward_grad[n][i][k]*jacobian_inverse[n][j];
                }
            }
        }
    }

}

template<int dim>
dealii::Tensor<2,dim,double> FEValuesShapeHessian<dim> :: shape_hessian_component(
        const unsigned int idof, 
        const unsigned int /*iquad*/, 
        const unsigned int istate, 
        const dealii::FESystem<dim,dim> &fe_ref)
{
    dealii::Tensor<1,dim,double> shape_grad_ref = fe_ref.shape_grad_component(idof, ref_point, istate);
    dealii::Tensor<2,dim,double> shape_hessian_ref = fe_ref.shape_grad_grad_component(idof, ref_point, istate);

    // Compute first term: J^{-T} \varphi_{\epsilon \epsilon} J^{-1}.
    // Using (A^T*B*A)_{i,j} = a_{ki} * b_{kl} * a_{lj} (in index notation).
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

    // Compute second term: \varphi_{\epsilon}^T * (J^{-1})_x
    // Using v^T*E_xx (i,j)  =  v_k (d^2 E_k/(dx_i dx_j) = v_k E_xx(k,i,j), with E_xx a third order tensor and v a vector.
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
