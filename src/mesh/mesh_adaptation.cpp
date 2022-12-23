#include "mesh_adaptation.h"
#include <deal.II/hp/refinement.h>

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshAdaptation<dim,real,MeshType>::MeshAdaptation(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, const Parameters::MeshAdaptationParam *const mesh_adaptation_param_input)
    : dg(dg_input)
    , current_mesh_adaptation_cycle(0)
    , mesh_adaptation_param(mesh_adaptation_param_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
        mesh_error = MeshErrorFactory<dim, 5, real, MeshType> :: create_mesh_error(dg);
    }


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::adapt_mesh()
{
    [[maybe_unused]] unsigned int expected_size_of_cellwise_errors = dg->triangulation->n_active_cells();
    cellwise_errors = mesh_error->compute_cellwise_errors();
    [[maybe_unused]] unsigned int actual_size_of_cellwise_errors = cellwise_errors.size();
    AssertDimension(expected_size_of_cellwise_errors, actual_size_of_cellwise_errors);

    fixed_fraction_isotropic_refinement_and_coarsening();
    current_mesh_adaptation_cycle++;
    pcout<<"Mesh has been adapted according to the specified error indicator. Adaptation cycle = "<<current_mesh_adaptation_cycle<<std::endl;
}


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::fixed_fraction_isotropic_refinement_and_coarsening()
{
    dealii::LinearAlgebra::distributed::Vector<real> old_solution(dg->solution);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<real>> solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    dg->high_order_grid->prepare_for_coarsening_and_refinement();

    if constexpr(dim == 1 || !std::is_same<MeshType, dealii::parallel::distributed::Triangulation<dim>>::value) 
    {
        dealii::GridRefinement::refine_and_coarsen_fixed_number(*(dg->high_order_grid->triangulation),
                                                                cellwise_errors,
                                                                mesh_adaptation_param->h_refine_fraction,
                                                                mesh_adaptation_param->h_coarsen_fraction);
    } 
    else 
    {
        dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(*(dg->high_order_grid->triangulation),
                                                                                        cellwise_errors,
                                                                                        mesh_adaptation_param->h_refine_fraction,
                                                                                        mesh_adaptation_param->h_coarsen_fraction);
    }

//=========================================================================================================================================================
    using MeshAdaptationTypeEnum = Parameters::MeshAdaptationParam::MeshAdaptationType;
    MeshAdaptationTypeEnum mesh_adaptation_type = mesh_adaptation_param->mesh_adaptation_type;


    if(mesh_adaptation_type == MeshAdaptationTypeEnum::h_adaptation){
        // Do nothing, cells are already flagged for h-adaptation
    } else if(mesh_adaptation_type == MeshAdaptationTypeEnum::p_adaptation){
        dealii::hp::Refinement::p_adaptivity_fixed_number(dg->dof_handler,
                                                          cellwise_errors,
                                                          1.0,
                                                          0.0);
        
        // If a cell is flagged for both h and p adaptation, perform only p adaptation.
        dealii::hp::Refinement::force_p_over_h(dg->dof_handler);
    } else if(mesh_adaptation_type == MeshAdaptationTypeEnum::hp_adaptation){
        smoothness_sensor_based_hp_refinement();
    }
//=========================================================================================================================================================

    dg->high_order_grid->triangulation->execute_coarsening_and_refinement();
    dg->high_order_grid->execute_coarsening_and_refinement();
    
    dg->allocate_system ();
    dg->solution.zero_out_ghost_values();
    solution_transfer.interpolate(dg->solution);
    dg->solution.update_ghost_values();
    dg->assemble_residual ();
}

template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::smoothness_sensor_based_hp_refinement()
{
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, update_flags); ///< FEValues of volume.

    std::vector< real > soln_coeff_high;
    std::vector<dealii::types::global_dof_index> dof_indices;

    for (auto cell : dg->dof_handler.active_cell_iterators()) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;
        if(!cell->refine_flag_set()) continue; 


        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;

        const dealii::FESystem<dim,dim> &fe_high = dg->fe_collection[i_fele];
        const unsigned int degree = fe_high.tensor_degree();

        if (degree == 0) 
        {
            pcout<<"Degree of the current cell is 0. Cannot compute smoothness indicator as we cannot interpolate to a lower polynomial order"<<std::endl;
            std::abort();
        }

        const unsigned int nstate = fe_high.components;
        const unsigned int n_dofs_high = fe_high.dofs_per_cell;

        fe_values_collection_volume.reinit (cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        dof_indices.resize(n_dofs_high);
        cell->get_dof_indices (dof_indices);

        soln_coeff_high.resize(n_dofs_high);
        for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
            soln_coeff_high[idof] = dg->solution[dof_indices[idof]];
        }

        // Lower degree basis.
        const unsigned int lower_degree = degree-1;
        const dealii::FE_DGQLegendre<dim> fe_dgq_lower(lower_degree);
        const dealii::FESystem<dim,dim> fe_lower(fe_dgq_lower, nstate);

        // Projection quadrature.
        const dealii::QGauss<dim> projection_quadrature(degree+5);
        std::vector< real > soln_coeff_lower = project_function<dim, real>( soln_coeff_high, fe_high, fe_lower, projection_quadrature);

        // Quadrature used for solution difference.
        const dealii::Quadrature<dim> &quadrature = fe_values_volume.get_quadrature();
        const std::vector<dealii::Point<dim,real>> &unit_quad_pts = quadrature.get_points();

        const unsigned int n_quad_pts = quadrature.size();
        const unsigned int n_dofs_lower = fe_lower.dofs_per_cell;

        real element_volume = 0.0;
        real error = 0.0;
        real soln_norm = 0.0;
        std::vector<real> soln_high(nstate);
        std::vector<real> soln_lower(nstate);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (unsigned int s=0; s<nstate; ++s) {
                soln_high[s] = 0.0;
                soln_lower[s] = 0.0;
            }
            // Interpolate solution
            for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
                  const unsigned int istate = fe_high.system_to_component_index(idof).first;
                  soln_high[istate] += soln_coeff_high[idof] * fe_high.shape_value_component(idof,unit_quad_pts[iquad],istate);
            }
            // Interpolate low order solution
            for (unsigned int idof=0; idof<n_dofs_lower; ++idof) {
                  const unsigned int istate = fe_lower.system_to_component_index(idof).first;
                  soln_lower[istate] += soln_coeff_lower[idof] * fe_lower.shape_value_component(idof,unit_quad_pts[iquad],istate);
            }
            // Quadrature
            element_volume += fe_values_volume.JxW(iquad);
            // Only integrate over the first state variable.
            for (unsigned int s=0; s<1/*nstate*/; ++s) 
            {
                error += (soln_high[s] - soln_lower[s]) * (soln_high[s] - soln_lower[s]) * fe_values_volume.JxW(iquad);
                soln_norm += soln_high[s] * soln_high[s] * fe_values_volume.JxW(iquad);
            }
        }

        if (soln_norm < 1e-12) 
        {
            continue;
        }
        
        real smoothness_sensor = error / soln_norm;
        
        if(smoothness_sensor < mesh_adaptation_param->hp_smoothness_tolerance)
        {
            cell->clear_refine_flag();
            cell->set_future_fe_index(cell->active_fe_index()+1);
        }
    }
}


template class MeshAdaptation<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
