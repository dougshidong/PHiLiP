#include "anisotropic_mesh_adaptation.h"
#include <deal.II/base/symmetric_tensor.h>

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: AnisotropicMeshAdaptation(
	std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, 
    const real _norm_Lp,
    const real _complexity,
	const bool _use_goal_oriented_approach)
	: dg(dg_input)
	, use_goal_oriented_approach(_use_goal_oriented_approach)
    , normLp(_norm_Lp)
    , complexity(_complexity)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0) 
{
    MPI_Comm_rank(mpi_communicator, &mpi_rank);
    MPI_Comm_size(mpi_communicator, &n_mpi);

    if(use_goal_oriented_approach)
    {
        if(normLp != 1)
        {
            pcout<<"Optimal metric for the goal oriented approach has been derived w.r.t the error in L1 norm. Aborting..."<<std::flush;
            std::abort();
        }
    }
}

template<int dim, int nstate, typename real, typename MeshType>
dealii::Tensor<2, dim, real> AnisotropicMeshAdaptation<dim, nstate, real, MeshType> 
    :: get_positive_definite_tensor(const dealii::Tensor<2, dim, real> &input_tensor) const
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

    dealii::Tensor<2, dim, real> positive_definite_tensor; // all entries are 0 by default.

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

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: initialize_cellwise_metric_and_hessians()
{
	cellwise_optimal_metric.clear();
	cellwise_hessian.clear();
	unsigned int n_active_cells = dg->triangulation->n_active_cells();
	
	dealii::Tensor<2, dim, real> zero_tensor; // initialized to 0 by default.
	for(unsigned int i=0; i<n_active_cells; ++i)
	{
		cellwise_optimal_metric.push_back(zero_tensor);
		cellwise_hessian.push_back(zero_tensor);
	}
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_optimal_metric()
{
   initialize_cellwise_metric_and_hessians();

}
} // PHiLiP namespace
