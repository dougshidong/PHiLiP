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
    dealii::SymmetricTensor<2,dim,real> symmetric_input_tensor(input_tensor); // Checks if input_tensor is symmetric in debug. It has to be symmetric because we are passing the Hessian.
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
        dealii::Tensor<2, dim, real> outer_product_i = dealii::outer_product(eigenvector_i, eigenvector_i);
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
	const unsigned int n_active_cells = dg->triangulation->n_active_cells();
	
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
	compute_abs_hessian(); // computes hessian according to goal oriented or feature based approach.

	const unsigned int n_active_cells = dg->triangulation->n_active_cells();
	dealii::Vector<real> abs_hessian_determinant(n_active_cells);
	
	// Compute hessian determinants
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		abs_hessian_determinant[cell_index] = dealii::determinant(cellwise_hessian[cell_index]);	
	}

	// Using Eq 4.40, page 153 in Dervieux, Alain, et al. Mesh Adaptation for Computational Fluid Dynamics 1. 2023.
	const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
	dealii::hp::MappingCollection<dim> mapping_collection(mapping);
	const dealii::UpdateFlags update_flags = dealii::update_JxW_values;
	dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, update_flags);
	
	real integral_val = 0;
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		const unsigned int i_fele = cell->active_fe_index();
		const unsigned int i_quad = i_fele;
		const unsigned int i_mapp = 0;
		fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
		const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

		const real exponent = normLp/(2.0*normLp + dim);
		const real integrand = pow(abs_hessian_determinant[cell_index], exponent);
		
		for(unsigned int iquad = 0; iquad<fe_values_volume.n_quadrature_points; ++iquad)
		{
			integral_val += integrand * fe_values_volume.JxW(iquad);
		}
	}

	const real integral_val_global = dealii::Utilities::MPI::sum(integral_val, mpi_communicator);
	const real exponent = 2.0/dim;
	const real scaling_factor = pow(complexity, exponent) * pow(integral_val_global, -exponent); // Factor by which the metric is scaled to get a mesh of required complexity/# of elements.

	// Now loop over to fill in the optimal metric.
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		
		const real exponent2 = -1.0/(2.0*normLp + dim);
		const real scaling_factor2 = pow(abs_hessian_determinant[cell_index], exponent2);
		const real scaling_factor_this_cell = scaling_factor * scaling_factor2;

		cellwise_optimal_metric[cell_index] = cellwise_hessian[cell_index];
		cellwise_optimal_metric[cell_index] *= scaling_factor_this_cell;
	}
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_abs_hessian()
{
	if(use_goal_oriented_approach)
	{
		compute_goal_oriented_hessian();
	}
	else
	{
		compute_feature_based_hessian();
	}

	// Get absolute values of the hessians (i.e. by taking abs of eigenvalues).
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		cellwise_hessian[cell_index] = get_positive_definite_tensor(cellwise_hessian[cell_index]);
	}

}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_feature_based_hessian()
{
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_goal_oriented_hessian()
{
}

// Instantiations
template class AnisotropicMeshAdaptation <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class AnisotropicMeshAdaptation <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class AnisotropicMeshAdaptation <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // PHiLiP namespace
