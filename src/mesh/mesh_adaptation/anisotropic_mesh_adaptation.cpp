#include "anisotropic_mesh_adaptation.h"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: AnisotropicMeshAdaptation(
	std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, 
	const bool _use_goal_oriented_approach)
	: dg(dg_input)
	, use_goal_oriented_approach(_use_goal_oriented_approach)
	{}

template<int dim, int nstate, typename real, typename MeshType>
dealii::SymmetricTensor<2, dim, real> AnisotropicMeshAdaptation<dim, nstate, real, MeshType> 
    :: get_positive_definite_tensor(const dealii::Tensor<2, dim, real> &input_tensor)
{
    const real min_eigenvalue = 1.0e-5;
    dealii::SymmetricTensor symmetric_input_tensor(input_tensor); // Checks if input_tensor is symmetric in debug. It has to be symmetric because we are passing the Hessian.
    std::array<std::pair<real, dealii::Tensor<1, dim, real>>, dim> eigen_pair = dealii::eigenvectors(symmetric_input_tensor);

    std::array<real, dim> abs_eignevalues;
    // Get absolute values of eigenvalues
    for(unsigned int i = 0; i<dim; ++i)
    {
        abs_eignevalues[i] = abs(eigen_pair[i].first);
        if(abs_eignevalues[i] < min_eigenvalue) {abs_eignevalues[i] = min_eigenvalue;}
    }

    dealii::SymmetricTensor<2, dim, real> positive_definite_tensor; // all entries are 0 by default.

    // Form the matrix again with the updated eigenvalues
    // If matrix of eigenvectors = [v1 v2 vdim], the new matrix would be
    // [v1 v2 vdim] * diag(eig1, eig2, eigdim) * [v1 v2 vdim]^T
    // = eig1*v1*v1^T + eig2*v2*v2^T + eigdim*vdim*vdim^T with vi*vi^T coming from dealii's outer product.
    for(unsigned int i=0; i<dim; ++i)
    {
        dealii::Tensor<1, dim, real> eigenvector_i = eigen_pair[i].second;
        dealii::Tensor<2, dim, real> outer_product_i = dealii::Tensor<2,dim,real>::outer_product(eigenvector_i, eigenvector_i);
        outer_product_i *= abs_eignevalues[i];
        positive_definite_tensor += outer_product_i;
    }

    return positive_definite_tensor;
}

} // PHiLiP namespace
