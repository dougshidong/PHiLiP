#include "anisotropic_mesh_adaptation.h"
#include <deal.II/base/symmetric_tensor.h>
#include "linear_solver/linear_solver.h"
#include "physics/physics_factory.h"
#include "physics/model_factory.h"
#include "mesh/gmsh_reader.hpp"
#include "metric_to_mesh_generator.h"
#include <deal.II/dofs/dof_renumbering.h>

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
        
        std::shared_ptr<Physics::ModelBase<dim,nstate,real>> pde_model_double    = Physics::ModelFactory<dim,nstate,real>::create_Model(dg->all_parameters);
        pde_physics_double  = Physics::PhysicsFactory<dim,nstate,real>::create_Physics(dg->all_parameters, pde_model_double);
		functional = FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(dg->all_parameters->functional_param, dg);
    }

	if(dg->get_min_fe_degree() != dg->get_max_fe_degree())
	{
		pcout<<"This class is currently coded assuming a constant poly degree. To be changed in future if required."<<std::endl;
		std::abort();
	}

	if(initial_poly_degree != 1)
	{
		pcout<<"Warning: The optimal metric used by this class has been derived for p1."
			 <<" For any other p, it might be a good approximation but will not be optimal."<<std::endl; 
	}

	Assert(this->dg->triangulation->get_mesh_smoothing() == typename dealii::Triangulation<dim>::MeshSmoothing(dealii::Triangulation<dim>::none),
           dealii::ExcMessage("Mesh smoothing might h-refine cells while changing p order."));

}

template<int dim, int nstate, typename real, typename MeshType>
dealii::Tensor<2, dim, real> AnisotropicMeshAdaptation<dim, nstate, real, MeshType> 
    :: get_positive_definite_tensor(const dealii::Tensor<2, dim, real> &input_tensor) const
{
    dealii::SymmetricTensor<2,dim,real> symmetric_input_tensor(input_tensor); // Checks if input_tensor is symmetric in debug. It has to be symmetric because we are passing the Hessian.
    std::array<std::pair<real, dealii::Tensor<1, dim, real>>, dim> eigen_pair = dealii::eigenvectors(symmetric_input_tensor);

    std::array<real, dim> abs_eignevalues;
	const real min_eigenvalue = 1.0e-8;
	const real max_eigenvalue = 1.0e8;
    // Get absolute values of eigenvalues
    for(unsigned int i = 0; i<dim; ++i)
    {
        abs_eignevalues[i] = abs(eigen_pair[i].first);
        if(abs_eignevalues[i] < min_eigenvalue) {abs_eignevalues[i] = min_eigenvalue;}
        if(abs_eignevalues[i] > max_eigenvalue) {abs_eignevalues[i] = max_eigenvalue;}
    }

    dealii::Tensor<2, dim, real> positive_definite_tensor; // all entries are 0 by default.

    // Form the matrix again with updated eigenvalues
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
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_cellwise_optimal_metric()
{
	initialize_cellwise_metric_and_hessians();
	compute_abs_hessian(); // computes hessian according to goal oriented or feature based approach.
    
    pcout<<"Computing optimal metric field."<<std::endl;

	const unsigned int n_active_cells = dg->triangulation->n_active_cells();
	dealii::Vector<real> abs_hessian_determinant(n_active_cells);
	
	// Compute hessian determinants
	for(const auto &cell : dg->dof_handler.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		
		const unsigned int cell_index = cell->active_cell_index();
		abs_hessian_determinant[cell_index] = dealii::determinant(cellwise_hessian[cell_index]);	
	}

	// Using Eq 4.40, page 153 in Dervieux, A. et al. Mesh Adaptation for Computational Fluid Dynamics 1. 2022.
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

	pcout<<"Done computing optimal metric."<<std::endl;
/*
	pcout<<"Cellwise metric = "<<std::endl;
    // Output metric
    for(const auto &cell : dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        const unsigned int cell_index = cell->active_cell_index();
        std::cout<<"cell index = "<<cell_index<<"  Processor# = "<<mpi_rank<<"\n"<<"Metric = "<<std::endl;
        for(unsigned int i = 0; i<dim; ++i)
        {
            for(unsigned int j=0; j<dim; ++j)
            {
                std::cout<<cellwise_optimal_metric[cell_index][i][j]<<" ";
            }
            std::cout<<std::endl;
        }
    }
*/
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
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: change_p_degree_and_interpolate_solution(const unsigned int poly_degree)
{
    pcout<<"Changing poly degree to "<<poly_degree<<" and interpolating solution."<<std::endl;
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
    pcout<<"Reconstructing p2 solution."<<std::endl;
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
	VectorType solution_old = dg->solution;
	solution_old.update_ghost_values();
	change_p_degree_and_interpolate_solution(2); // Interpolate to p2
	reconstruct_p2_solution();
    
	pcout<<"Computing feature based Hessian."<<std::endl;
	// Compute Hessian of the solution for now (can be changed to Mach number or some other sensor when required).
	//const auto mapping = (*(dg->high_order_grid->mapping_fe_field)); // CHANGE IT BACK
	dealii::MappingQGeneric<dim, dim> mapping(dg->high_order_grid->dof_handler_grid.get_fe().degree);
	dealii::hp::MappingCollection<dim> mapping_collection(mapping);
	const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients | dealii::update_hessians | dealii::update_quadrature_points | dealii::update_JxW_values;
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
	
        cellwise_hessian[cell_index] = 0;
		const unsigned int iquad = get_iquad_near_cellcenter(fe_values_volume.get_quadrature());
		
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
		{
			const unsigned int icomp = fe_values_volume.get_fe().system_to_component_index(idof).first;
			// Computing hesssian of solution at state 0. Might need to change it later.
            if(icomp == 0)
            {
			    cellwise_hessian[cell_index] += dg->solution(dof_indices[idof])*fe_values_volume.shape_hessian_component(idof, iquad, icomp); 
            }
		}
		cellwise_hessian[cell_index] = get_positive_definite_tensor(cellwise_hessian[cell_index]);
	}
	
	change_p_degree_and_interpolate_solution(initial_poly_degree);
	dg->solution = solution_old; // reset solution
    dg->solution.update_ghost_values();
}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: compute_goal_oriented_hessian()
{
    // Compute the adjoint ===================================================================
    VectorType adjoint(dg->solution); 
    dg->assemble_residual(true);
    functional->evaluate_functional(true);
    solve_linear(dg->system_matrix_transpose, functional->dIdw, adjoint, dg->all_parameters->linear_solver_param);
    adjoint *= -1.0;
    adjoint.update_ghost_values();
    //==========================================================================================
	// Compute adjoint gradient
	std::vector<std::array<dealii::Tensor<1, dim, real>, nstate>> adjoint_gradient(dg->triangulation->n_active_cells());
	{
		const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
		dealii::hp::MappingCollection<dim> mapping_collection(mapping);
		const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
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
			const unsigned int iquad = get_iquad_near_cellcenter(fe_values_volume.get_quadrature());
			for(unsigned int istate = 0; istate<nstate; ++istate)
			{
				adjoint_gradient[cell_index][istate] = 0;
			}
			for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
			{ 
				const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
				adjoint_gradient[cell_index][istate] += adjoint(dof_indices[idof])*fe_values_volume.shape_grad_component(idof, iquad, istate);
			}
		} // cell loop ends
	}
	//=========================================================================================
	dg->solution.update_ghost_values();
	const VectorType solution_old = dg->solution;
	change_p_degree_and_interpolate_solution(2); // Interpolate to p2
	reconstruct_p2_solution();
  
	pcout<<"Computing goal-oriented Hessian."<<std::endl;
    // Compute goal oriented pseudo hessian.
    // From Eq. 28 in Loseille, A., Dervieux, A., and Alauzet, F. "Fully anisotropic goal-oriented mesh adaptation for 3D steady Euler equations.", 2010.
    // Also, as suggested in the footnote on page 78 in Dervieux, A. et al. Mesh Adaptation for Computational Fluid Dynamics 2. 2022, metric terms related to 
    // faces do not make a difference and hence are not included.
	//const auto mapping = (*(dg->high_order_grid->mapping_fe_field)); // CHANGE IT BACK
	dealii::MappingQGeneric<dim, dim> mapping(dg->high_order_grid->dof_handler_grid.get_fe().degree);
	dealii::hp::MappingCollection<dim> mapping_collection(mapping);
	const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients | dealii::update_hessians | dealii::update_quadrature_points | dealii::update_JxW_values;
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

		const unsigned int iquad = get_iquad_near_cellcenter(fe_values_volume.get_quadrature());

        // Obtain flux coeffs
        std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> flux_coeffs(n_dofs_cell);
        get_flux_coeffs(flux_coeffs, fe_values_volume, dof_indices, cell);
        
        // Compute Hessian
        cellwise_hessian[cell_index] = 0;
        for(unsigned int istate = 0; istate < nstate; ++istate)
        {
            for(unsigned int idim = 0; idim < dim; ++idim)
            {
                dealii::Tensor<2,dim,real> flux_hessian_at_istate_idim;
                for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
                {
                    const unsigned int icomp = fe_values_volume.get_fe().system_to_component_index(idof).first;
                    if(icomp == istate)
                    {
                        flux_hessian_at_istate_idim += flux_coeffs[idof][istate][idim]*fe_values_volume.shape_hessian_component(idof, iquad, icomp);
                    }
                } // idof
                flux_hessian_at_istate_idim = get_positive_definite_tensor(flux_hessian_at_istate_idim);
                flux_hessian_at_istate_idim *= abs(adjoint_gradient[cell_index][istate][idim]);
                cellwise_hessian[cell_index] += flux_hessian_at_istate_idim;
            } //idim
        } //istate
    } // cell loop ends

	change_p_degree_and_interpolate_solution(initial_poly_degree);
	dg->solution = solution_old; // reset solution
    dg->solution.update_ghost_values();
}

template<int dim, int nstate, typename real, typename MeshType>
unsigned int AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: get_iquad_near_cellcenter(
	const dealii::Quadrature<dim> &volume_quadrature) const
{
    dealii::Point<dim,real> ref_center;
    for(unsigned int idim =0; idim < dim; ++idim) 
        {ref_center[idim] = 0.5;}

    unsigned int iquad_center = 0;
    real min_distance = 10000.0;
    for(unsigned int iquad = 0; iquad<volume_quadrature.size(); ++iquad)
    {
        const dealii::Point<dim, real> &ref_point = volume_quadrature.point(iquad);
        const real ref_distance = ref_point.distance(ref_center); 
        if(min_distance > ref_distance)
        {
            min_distance = ref_distance;
            iquad_center = iquad;
        }
    }

    return iquad_center;
}

// Flux referenced by flux[idof][istate][idim]
template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: get_flux_coeffs(
	std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> &flux_coeffs, 
	const dealii::FEValues<dim,dim> &fe_values_input,
	const std::vector<dealii::types::global_dof_index> &dof_indices,
	typename dealii::DoFHandler<dim>::active_cell_iterator cell) const
{
    if( ! fe_values_input.get_fe().has_support_points() )
    {
        pcout<<"The code here treats flux at support points as flux coeff at idof "
             <<"which requires an interpolatory FE with support points. Aborting.."<<std::endl;
        std::abort();
    }
        
    const unsigned int n_dofs_cell = dof_indices.size();
    dealii::Quadrature<dim> support_pts = fe_values_input.get_fe().get_unit_support_points();
    dealii::FEValues<dim, dim> fe_values_support_pts(fe_values_input.get_mapping(), fe_values_input.get_fe(),
                                                     support_pts, dealii::update_values);
    fe_values_support_pts.reinit(cell);
    const unsigned int n_quad_pts = fe_values_support_pts.n_quadrature_points;
    Assert(n_quad_pts == n_dofs_cell, dealii::ExcMessage("n_quad_pts != n_dofs_cell"));

    for(unsigned int iquad = 0; iquad<n_quad_pts; ++iquad)
    {
        std::array< real, nstate > soln_at_q;
		soln_at_q.fill(0.0);
        for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
        {
            const unsigned int istate = fe_values_support_pts.get_fe().system_to_component_index(idof).first;
            soln_at_q[istate] += dg->solution(dof_indices[idof]) * fe_values_support_pts.shape_value_component(idof, iquad, istate);
        }
        
        // Here we assume flux_coeffs[idof] = flux_at_iquad.
        flux_coeffs[iquad] = pde_physics_double->convective_flux(soln_at_q);
    }

}

template<int dim, int nstate, typename real, typename MeshType>
void AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: adapt_mesh()
{
	compute_cellwise_optimal_metric();
	
	std::unique_ptr<MetricToMeshGenerator<dim, nstate, real>> metric_to_mesh_generator
		= std::make_unique<MetricToMeshGenerator<dim, nstate, real>> (dg->high_order_grid->mapping_fe_field, dg->triangulation);
	metric_to_mesh_generator->generate_mesh_from_cellwise_metric(cellwise_optimal_metric);
	
	std::shared_ptr<HighOrderGrid<dim,double,MeshType>> new_high_order_mesh = read_gmsh <dim, dim> (
																	metric_to_mesh_generator->get_generated_mesh_filename());
	dg->set_high_order_grid(new_high_order_mesh);
	dg->allocate_system();

	// Need to either interpolate or initialize solution on the new mesh. Currently just set it to 0.
	dg->solution = 0;
	dg->solution.update_ghost_values();

	metric_to_mesh_generator->delete_generated_files();
}

// Instantiations
#if PHILIP_DIM!=1
template class AnisotropicMeshAdaptation <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AnisotropicMeshAdaptation <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // PHiLiP namespace
