#include "anisotropic_mesh_adaptation.h"
#include <deal.II/base/symmetric_tensor.h>
#include "linear_solver/linear_solver.h"

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
	, initial_poly_degree(dg->get_min_fe_degree())
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
	if(dg->get_min_fe_degree() != dg->get_max_fe_degree())
	{
		pcout<<"This class is currently coded assuming a constant poly degree. To be changed in future if required."<<std::endl;
		std::abort();
	}
	if(initial_poly_degree != 1)
	{
		pcout<<"Warning: The optimal metric used by this class has been derived for p1."
			 <<" For any other p, it might be a good apprximation but will not not optimal"<<std::endl; 
	}

	Assert(this->dg->triangulation->get_mesh_smoothing() == typename dealii::Triangulation<dim>::MeshSmoothing(dealii::Triangulation<dim>::none),
           dealii::ExcMessage("Mesh smoothing might h-refine cells while changing p order."));
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
	VectorType solution_old = dg->solution;
	solution_old.update_ghost_values();
	change_p_degree_and_interpolate_solution(2); // Change to p2
	reconstruct_p2_solution();
	if(use_goal_oriented_approach)
	{
		compute_goal_oriented_hessian();
	}
	else
	{
		compute_feature_based_hessian();
	}
	change_p_degree_and_interpolate_solution(initial_poly_degree);
	dg->solution = solution_old; // reset solution

	// Get absolute values of the hessians (i.e. by taking abs of eigenvalues).
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		cellwise_hessian[cell_index] = get_positive_definite_tensor(cellwise_hessian[cell_index]);
	}

}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: change_p_degree_and_interpolate_solution(const unsigned int poly_degree)
{
	VectorType solution_coarse = dg->solution;
	solution_coarse.update_ghost_values();

	using DoFHandlerType   = typename dealii::DoFHandler<dim>;
	using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;
	
	SolutionTransfer solution_transfer(this->dg->dof_handler);
	solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);
	
	dg->set_all_cells_fe_degree(poly_degree);
	dg->allocate_system();
	dg->solution.zero_out_ghosts();
	
	if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>,
                                 decltype(solution_transfer)>) {
        solution_transfer.interpolate(solution_coarse, this->dg->solution);
    } else {
        solution_transfer.interpolate(this->dg->solution);
    }

    this->dg->solution.update_ghost_values();
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: reconstruct_p2_solution()
{
	assert(dg->get_min_fe_degree() == 2);
	dg->assemble_residual(true);
	VectorType delU = dg->solution;
	solve_linear(dg->system_matrix, dg->right_hand_side, delU, dg->all_parameters->linear_solver_param);
	delU *= -1.0;
	delU.update_ghost_values();
	dg->solution += delU;
	dg->solution.update_ghost_values();
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_feature_based_hessian()
{
	// Compute Hessian of the solution at state 0 for now. It can be changed to Mach number or some other sensor later if required.
	//const auto mapping = (*(dg->high_order_grid->mapping_fe_field)); // CHANGE IT BACK
	dealii::MappingQGeneric<dim, dim> mapping(dg->high_order_grid->dof_handler_grid.get_fe().degree);
	dealii::hp::MappingCollection<dim> mapping_collection(mapping);
	const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients | dealii::update_hessians | dealii::update_quadrature_points | dealii::update_JxW_values
        | dealii::update_inverse_jacobians;
	dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, update_flags);
	
	std::vector<dealii::types::global_dof_index> dof_indices;

	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		const unsigned int i_fele = cell->active_fe_index();
		const unsigned int i_quad = i_fele;
		const unsigned int i_mapp = 0;
		fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
		const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
		
		const unsigned int n_dofs_cell = fe_values_volume.dofs_per_cell; 
		dof_indices.resize(n_dofs_cell);
		cell->get_dof_indices(dof_indices);
		
		// Since Hessian is constant in the cell for a p2 solution, we compute it at just one quadrature point.
		unsigned int iquad = fe_values_volume.n_quadrature_points/2;
		for(unsigned int idof = 0; idof<n_dofs_cell; ++iquad)
		{
			const unsigned int icomp = fe_values_volume.get_fe().system_to_component_index(idof).first;
			// Adding hesssians of all components. Might need to change it later as required.
			cellwise_hessian[cell_index] += dg->solution(dof_indices[idof])*fe_values_volume.shape_hessian_component(idof, iquad, icomp); 
		}
	}
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
