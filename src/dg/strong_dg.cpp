#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

#include "strong_dg.hpp"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::DGStrong(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
    , do_compute_filtered_solution(this->all_parameters->physics_model_param.do_compute_filtered_solution)
    , apply_modal_high_pass_filter_on_filtered_solution(this->all_parameters->physics_model_param.apply_modal_high_pass_filter_on_filtered_solution)
    , poly_degree_max_large_scales(this->all_parameters->physics_model_param.poly_degree_max_large_scales)
{ }

/***********************************************************
*
*       Build operators and solve for RHS
*
***********************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index                  current_cell_index,
    const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
    const unsigned int                                     poly_degree,
    const unsigned int                                     grid_degree,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis,
    OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
    OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
    std::array<std::vector<real>,dim>                      &mapping_support_points,
    dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume*/,
    dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume_lagrange*/,
    const dealii::FESystem<dim,dim>                        &/*current_fe_ref*/,
    dealii::Vector<real>                                   &local_rhs_int_cell,
    std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
    const bool                                             compute_auxiliary_right_hand_side,
    const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/)
{
    // Check if the current cell's poly degree etc is different then previous cell's.
    // If the current cell's poly degree is different, then we recompute the 1D 
    // polynomial basis functions. Otherwise, we use the previous values in reference space.
    if(poly_degree != soln_basis.current_degree){
        soln_basis.current_degree = poly_degree; 
        flux_basis.current_degree = poly_degree; 
        mapping_basis.current_degree  = poly_degree; 
        this->reinit_operators_for_cell_residual_loop(poly_degree, poly_degree, grid_degree, 
                                                      soln_basis, soln_basis, 
                                                      flux_basis, flux_basis, 
                                                      flux_basis_stiffness, 
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //Rewrite the high_order_grid->volume_nodes in a way we can use sum-factorization on.
    //That is, splitting up the vector by the dimension.
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_grid_nodes);
    }
    const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
        const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
        const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
        const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
        const unsigned int igrid_node = index_renumbering[ishape];
        mapping_support_points[istate][igrid_node] = val; 
    }

    //build the volume metric cofactor matrix and the determinant of the volume metric Jacobian
    //Also, computes the physical volume flux nodes if needed from flag passed to constructor in dg.cpp
    metric_oper.build_volume_metric_operators(
        this->volume_quadrature_collection[poly_degree].size(), n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    if(compute_auxiliary_right_hand_side){
        assemble_volume_term_auxiliary_equation (
            cell_dofs_indices,
            poly_degree,
            soln_basis,
            flux_basis,
            metric_oper,
            local_auxiliary_RHS);
    }
    else{
        assemble_volume_term_strong(
            cell,
            current_cell_index,
            cell_dofs_indices,
            poly_degree,
            soln_basis,
            flux_basis,
            flux_basis_stiffness,
            soln_basis_projection_oper_int,
            metric_oper,
            local_rhs_int_cell);
    }
}
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index                  current_cell_index,
    const unsigned int                                     iface,
    const unsigned int                                     boundary_id,
    const real                                             penalty,
    const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &/*metric_dof_indices*/,
    const unsigned int                                     poly_degree,
    const unsigned int                                     /*grid_degree*/,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis,
    OPERATOR::local_basis_stiffness<dim,2*dim,real>        &/*flux_basis_stiffness*/,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &/*soln_basis_projection_oper_ext*/,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
    OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
    std::array<std::vector<real>,dim>                      &mapping_support_points,
    dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_int*/,
    const dealii::FESystem<dim,dim>                        &/*current_fe_ref*/,
    dealii::Vector<real>                                   &local_rhs_int_cell,
    std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
    const bool                                             compute_auxiliary_right_hand_side,
    const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/)
{

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper.build_facet_metric_operators(
        iface,
        this->face_quadrature_collection[poly_degree].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    if(compute_auxiliary_right_hand_side){
        assemble_boundary_term_auxiliary_equation (
            iface, current_cell_index, poly_degree,
            boundary_id, cell_dofs_indices, 
            soln_basis, metric_oper,
            local_auxiliary_RHS);
    }
    else{
        assemble_boundary_term_strong (
            iface,
            current_cell_index,
            boundary_id, poly_degree, penalty, 
            cell_dofs_indices, 
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            local_rhs_int_cell);
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
    const dealii::types::global_dof_index                  current_cell_index,
    const dealii::types::global_dof_index                  neighbor_cell_index,
    const unsigned int                                     iface,
    const unsigned int                                     neighbor_iface,
    const real                                             penalty,
    const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &/*current_metric_dofs_indices*/,
    const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
    const unsigned int                                     poly_degree_int,
    const unsigned int                                     poly_degree_ext,
    const unsigned int                                     /*grid_degree_int*/,
    const unsigned int                                     grid_degree_ext,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_ext,
    OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
    OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
    std::array<std::vector<real>,dim>                      &mapping_support_points,
    dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_int*/,
    dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_ext*/,
    dealii::Vector<real>                                   &current_cell_rhs,
    dealii::Vector<real>                                   &neighbor_cell_rhs,
    std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
    dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
    std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
    const bool                                             compute_auxiliary_right_hand_side,
    const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/)
{

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper_int.build_facet_metric_operators(
        iface,
        this->face_quadrature_collection[poly_degree_int].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    if(poly_degree_ext != soln_basis_ext.current_degree){
        soln_basis_ext.current_degree    = poly_degree_ext; 
        flux_basis_ext.current_degree    = poly_degree_ext; 
        mapping_basis.current_degree     = poly_degree_ext; 
        this->reinit_operators_for_cell_residual_loop(poly_degree_int, poly_degree_ext, grid_degree_ext, 
                                                      soln_basis_int, soln_basis_ext, 
                                                      flux_basis_int, flux_basis_ext, 
                                                      flux_basis_stiffness, 
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }

    if(!compute_auxiliary_right_hand_side){//only for primary equations
        //get neighbor metric operator
        //rewrite the high_order_grid->volume_nodes in a way we can use sum-factorization on.
        //that is, splitting up the vector by the dimension.
        std::array<std::vector<real>,dim> mapping_support_points_neigh;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points_neigh[idim].resize(n_grid_nodes);
        }
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree_ext);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->high_order_grid->volume_nodes[neighbor_metric_dofs_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points_neigh[istate][igrid_node] = val; 
        }
        //build the metric operators for strong form
        metric_oper_ext.build_volume_metric_operators(
            this->volume_quadrature_collection[poly_degree_ext].size(), n_grid_nodes,
            mapping_support_points_neigh,
            mapping_basis,
            this->all_parameters->use_invariant_curl_form);
    }

    if(compute_auxiliary_right_hand_side){
        const unsigned int n_dofs_neigh_cell = this->fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
        std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs_aux (n_dofs_neigh_cell ); // defaults to 0.0 initialization
        assemble_face_term_auxiliary_equation (
            iface, neighbor_iface, 
            current_cell_index, neighbor_cell_index,
            poly_degree_int, poly_degree_ext,
            current_dofs_indices, neighbor_dofs_indices,
            soln_basis_int, soln_basis_ext,
            metric_oper_int,
            current_cell_rhs_aux, neighbor_cell_rhs_aux);
        // add local contribution from neighbor cell to global vector
        for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
            for(int idim=0; idim<dim; idim++){
                rhs_aux[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs_aux[i][idim];
            }
        }
    }
    else{
        assemble_face_term_strong (
            iface, neighbor_iface, 
            current_cell_index,
            neighbor_cell_index,
            poly_degree_int, poly_degree_ext,
            penalty,
            current_dofs_indices, neighbor_dofs_indices,
            soln_basis_int, soln_basis_ext,
            flux_basis_int, flux_basis_ext,
            soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
            metric_oper_int, metric_oper_ext,
            current_cell_rhs, neighbor_cell_rhs);
        // add local contribution from neighbor cell to global vector
        const unsigned int n_dofs_neigh_cell = this->fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
        for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
            rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
        }
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_subface_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
    const dealii::types::global_dof_index                  current_cell_index,
    const dealii::types::global_dof_index                  neighbor_cell_index,
    const unsigned int                                     iface,
    const unsigned int                                     neighbor_iface,
    const unsigned int                                     /*neighbor_i_subface*/,
    const real                                             penalty,
    const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
    const unsigned int                                     poly_degree_int,
    const unsigned int                                     poly_degree_ext,
    const unsigned int                                     grid_degree_int,
    const unsigned int                                     grid_degree_ext,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_ext,
    OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
    OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
    std::array<std::vector<real>,dim>                      &mapping_support_points,
    dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
    dealii::hp::FESubfaceValues<dim,dim>                   &/*fe_values_collection_subface*/,
    dealii::Vector<real>                                   &current_cell_rhs,
    dealii::Vector<real>                                   &neighbor_cell_rhs,
    std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
    dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
    std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
    const bool                                             compute_auxiliary_right_hand_side,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    assemble_face_term_and_build_operators(
        cell,
        neighbor_cell,
        current_cell_index,
        neighbor_cell_index,
        iface,
        neighbor_iface,
        penalty,
        current_dofs_indices,
        neighbor_dofs_indices,
        current_metric_dofs_indices,
        neighbor_metric_dofs_indices,
        poly_degree_int,
        poly_degree_ext,
        grid_degree_int,
        grid_degree_ext,
        soln_basis_int,
        soln_basis_ext,
        flux_basis_int,
        flux_basis_ext,
        flux_basis_stiffness,
        soln_basis_projection_oper_int,
        soln_basis_projection_oper_ext,
        metric_oper_int,
        metric_oper_ext,
        mapping_basis,
        mapping_support_points,
        fe_values_collection_face_int,
        fe_values_collection_face_int,
        current_cell_rhs,
        neighbor_cell_rhs,
        current_cell_rhs_aux,
        rhs,
        rhs_aux,
        compute_auxiliary_right_hand_side,
        compute_dRdW, compute_dRdX, compute_d2R);

}
/*******************************************************************
 *
 *
 *                      AUXILIARY EQUATIONS
 *
 *
 *******************************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_auxiliary_residual()
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using ODE_enum = Parameters::ODESolverParam::ODESolverEnum;
    const PDE_enum pde_type = this->all_parameters->pde_type;

    if(pde_type == PDE_enum::burgers_viscous){
        pcout << "DG Strong not yet verified for Burgers' viscous. Aborting..." << std::endl;
        std::abort();
    }
    // NOTE: auxiliary currently only works explicit time advancement - not implicit
    if (this->use_auxiliary_eq && !(this->all_parameters->ode_solver_param.ode_solver_type == ODE_enum::implicit_solver)) {
        //set auxiliary rhs to 0
        for(int idim=0; idim<dim; idim++){
            this->auxiliary_right_hand_side[idim] = 0;
        }
        //initialize this to use DG cell residual loop. Note, FEValues to be deprecated in future.
        const auto mapping = (*(this->high_order_grid->mapping_fe_field));

        dealii::hp::MappingCollection<dim> mapping_collection(mapping);

        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, this->fe_collection, this->volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
        dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.
         
        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, this->fe_collection_lagrange, this->volume_quadrature_collection, this->volume_update_flags);

        OPERATOR::basis_functions<dim,2*dim,real> soln_basis_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim,real> soln_basis_ext(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim,real> flux_basis_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim,real> flux_basis_ext(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::local_basis_stiffness<dim,2*dim,real> flux_basis_stiffness(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::mapping_shape_functions<dim,2*dim,real> mapping_basis(1, this->max_grid_degree, this->max_grid_degree);
        OPERATOR::vol_projection_operator<dim,2*dim,real> soln_basis_projection_oper_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::vol_projection_operator<dim,2*dim,real> soln_basis_projection_oper_ext(1, this->max_degree, this->max_grid_degree); 
         
        this->reinit_operators_for_cell_residual_loop(
            this->max_degree, this->max_degree, this->max_grid_degree, 
            soln_basis_int, soln_basis_ext, 
            flux_basis_int, flux_basis_ext, 
            flux_basis_stiffness, 
            soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
            mapping_basis);

        //loop over cells solving for auxiliary rhs
        auto metric_cell = this->high_order_grid->dof_handler_grid.begin_active();
        for (auto soln_cell = this->dof_handler.begin_active(); soln_cell != this->dof_handler.end(); ++soln_cell, ++metric_cell) {
            if (!soln_cell->is_locally_owned()) continue;

            this->assemble_cell_residual (
                soln_cell,
                metric_cell,
                false, false, false,
                fe_values_collection_volume,
                fe_values_collection_face_int,
                fe_values_collection_face_ext,
                fe_values_collection_subface,
                fe_values_collection_volume_lagrange,
                soln_basis_int,
                soln_basis_ext,
                flux_basis_int,
                flux_basis_ext,
                flux_basis_stiffness,
                soln_basis_projection_oper_int, 
                soln_basis_projection_oper_ext,
                mapping_basis,
                true,
                this->right_hand_side,
                this->auxiliary_right_hand_side);
        } // end of cell loop

        for(int idim=0; idim<dim; idim++){
            //compress auxiliary rhs for solution transfer across mpi ranks
            this->auxiliary_right_hand_side[idim].compress(dealii::VectorOperation::add);
            //update ghost values
            this->auxiliary_right_hand_side[idim].update_ghost_values();

            //solve for auxiliary solution for each dimension
            if(this->all_parameters->use_inverse_mass_on_the_fly)
                this->apply_inverse_global_mass_matrix(this->auxiliary_right_hand_side[idim], this->auxiliary_solution[idim], true);
            else
                this->global_inverse_mass_matrix_auxiliary.vmult(this->auxiliary_solution[idim], this->auxiliary_right_hand_side[idim]);

            //update ghost values of auxiliary solution
            this->auxiliary_solution[idim].update_ghost_values();
        }
    }//end of if statement for diffusive
    else if (this->use_auxiliary_eq && (this->all_parameters->ode_solver_param.ode_solver_type == ODE_enum::implicit_solver)) {
        pcout << "ERROR: " << "auxiliary currently only works for explicit time advancement. Aborting..." << std::endl;
        std::abort();
    } else {
        // Do nothing
    }
}

/**************************************************
 *
 *         AUXILIARY RESIDUAL FUNCTIONS
 *
 **************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_auxiliary_equation(
    const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
    const unsigned int poly_degree,
    OPERATOR::basis_functions<dim,2*dim,real> &soln_basis,
    OPERATOR::basis_functions<dim,2*dim,real> &flux_basis,
    OPERATOR::metric_operators<real,dim,2*dim> &metric_oper,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS)
{
    //Please see header file for exact formula we are solving.
    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const std::vector<double> &quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();

    //Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    //We immediately separate them by state as to be able to use sum-factorization
    //in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    //mult would sum the states at the quadrature point.
    //That is why the basis functions are of derived class state rather than base.
    std::array<std::vector<real>,nstate> soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);

        soln_coeff[istate][ishape] = this->solution(current_dofs_indices[idof]);
    }
    //Interpolate each state to the quadrature points using sum-factorization
    //with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        std::vector<real> soln_at_q(n_quad_pts);
        //interpolate soln coeff to volume cubature nodes
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q,
                                         soln_basis.oneD_vol_operator);
        //the volume integral for the auxiliary equation is the physical integral of the physical gradient of the solution.
        //That is, we need to physically integrate (we have determinant of Jacobian cancel) the Eq. (12) (with u for chi) in
        //Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.

        //apply gradient of reference basis functions on the solution at volume cubature nodes
        dealii::Tensor<1,dim,std::vector<real>> ref_gradient_basis_fns_times_soln;
        for(int idim=0; idim<dim; idim++){
            ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
        }
        flux_basis.gradient_matrix_vector_mult_1D(soln_at_q, ref_gradient_basis_fns_times_soln,
                                                  flux_basis.oneD_vol_operator,
                                                  flux_basis.oneD_grad_operator);
        //transform the gradient into a physical gradient operator scaled by determinant of metric Jacobian
        //then apply the inner product in each direction
        for(int idim=0; idim<dim; idim++){
            std::vector<real> phys_gradient_u(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(int jdim=0; jdim<dim; jdim++){
                    //transform into the physical gradient
                    phys_gradient_u[iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                 * ref_gradient_basis_fns_times_soln[jdim][iquad];
                }
            }
            //Note that we let the determiant of the metric Jacobian cancel off between the integral and physical gradient
            std::vector<real> rhs(n_shape_fns);
            soln_basis.inner_product_1D(phys_gradient_u, quad_weights,
                                        rhs,
                                        soln_basis.oneD_vol_operator,
                                        false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            //write the the auxiliary rhs for the test function.
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                local_auxiliary_RHS[istate*n_shape_fns + ishape][idim] += rhs[ishape];
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_auxiliary_equation(
    const unsigned int iface,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int poly_degree,
    const unsigned int boundary_id,
    const std::vector<dealii::types::global_dof_index> &dofs_indices,
    OPERATOR::basis_functions<dim,2*dim,real> &soln_basis,
    OPERATOR::metric_operators<real,dim,2*dim> &metric_oper,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS)
{
    (void) current_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs          = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns     = n_dofs / nstate;
    AssertDimension (n_dofs, dofs_indices.size());

    //Extract interior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        //allocate
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);
        //solve
        soln_coeff[istate][ishape] = this->solution(dofs_indices[idof]);
    }

    //Interpolate soln to facet, and gradient to facet.
    std::array<std::vector<real>,nstate> soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> ref_grad_soln_at_vol_q;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface, soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
        //solve reference gradient of soln at facet cubature nodes
        for(int idim=0; idim<dim; idim++){
            ref_grad_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
        }
        soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_grad_soln_at_vol_q[istate],
                                                  soln_basis.oneD_vol_operator,
                                                  soln_basis.oneD_grad_operator);
    }

    // Get physical gradient of solution on the surface
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> phys_grad_soln_at_surf_q;
    for(int istate=0; istate<nstate; istate++){
        //transform the gradient into a physical gradient operator
        for(int idim=0; idim<dim; idim++){
            std::vector<real> phys_gradient_u(n_quad_pts_vol);
            for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
                for(int jdim=0; jdim<dim; jdim++){
                    //transform into the physical gradient
                    phys_gradient_u[iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                 * ref_grad_soln_at_vol_q[istate][jdim][iquad];
                }
                phys_gradient_u[iquad] /= metric_oper.det_Jac_vol[iquad];
            }
            phys_grad_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            //interpolate physical volume gradient of the solution to the surface
            soln_basis.matrix_vector_mult_surface_1D(iface, phys_gradient_u, phys_grad_soln_at_surf_q[istate][idim],
                                                     soln_basis.oneD_surf_operator,
                                                     soln_basis.oneD_vol_operator);
        }
    }

    //evaluate physical facet fluxes dot product with physical unit normal scaled by determinant of metric facet Jacobian
    //the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> surf_num_flux_minus_surf_soln_dot_normal;
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        //Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        //The way it is stored in metric_operators is to use sum-factorization in each direction,
        //but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        //Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        //This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> phys_grad_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_surf_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                phys_grad_soln_state[istate][idim] = phys_grad_soln_at_surf_q[istate][idim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper.transform_reference_to_physical(unit_ref_normal_int,
                                                    metric_cofactor_surf,
                                                    unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        std::array<real,nstate> soln_boundary;
        std::array<dealii::Tensor<1,dim,real>,nstate> grad_soln_boundary;
        dealii::Point<dim,real> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
        }
        this->pde_physics_double->boundary_face_values (boundary_id, surf_flux_node, unit_phys_normal_int, soln_state, phys_grad_soln_state, soln_boundary, grad_soln_boundary);

        std::array<real,nstate> diss_soln_num_flux;
        diss_soln_num_flux = this->diss_num_flux_double->evaluate_solution_flux(soln_state, soln_boundary, unit_phys_normal_int);

        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    surf_num_flux_minus_surf_soln_dot_normal[istate][idim].resize(n_face_quad_pts);
                }
                //solve
                surf_num_flux_minus_surf_soln_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q[istate][iquad]) * unit_phys_normal_int[idim] * face_Jac_norm_scaled;
            }
        }
    }
    //solve residual and set
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();
    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            std::vector<real> rhs(n_shape_fns);

            soln_basis.inner_product_surface_1D(iface, 
                                                surf_num_flux_minus_surf_soln_dot_normal[istate][idim],
                                                surf_quad_weights, rhs,
                                                soln_basis.oneD_surf_operator,
                                                soln_basis.oneD_vol_operator,
                                                false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                local_auxiliary_RHS[istate*n_shape_fns + ishape][idim] += rhs[ishape]; 
            }
        }
    }
}
/*********************************************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_auxiliary_equation(
    const unsigned int iface, const unsigned int neighbor_iface,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int poly_degree_int, 
    const unsigned int poly_degree_ext,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    OPERATOR::basis_functions<dim,2*dim,real> &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real> &soln_basis_ext,
    OPERATOR::metric_operators<real,dim,2*dim> &metric_oper_int,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS_int,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS_ext)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    //Extract interior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_int;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_int].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_int].system_to_component_index(idof).second;
        //allocate
        if(ishape == 0) soln_coeff_int[istate].resize(n_shape_fns_int);

        //solve
        soln_coeff_int[istate][ishape] = this->solution(dof_indices_int[idof]);
    }

    //Extract exterior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_ext;
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_ext].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_ext].system_to_component_index(idof).second;
        //allocate
        if(ishape == 0) soln_coeff_ext[istate].resize(n_shape_fns_ext);

        //solve
        soln_coeff_ext[istate][ishape] = this->solution(dof_indices_ext[idof]);
    }

    //Interpolate soln modal coefficients to the facet
    std::array<std::vector<real>,nstate> soln_at_surf_q_int;
    std::array<std::vector<real>,nstate> soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //evaluate physical facet fluxes dot product with physical unit normal scaled by determinant of metric facet Jacobian
    //the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> surf_num_flux_minus_surf_soln_int_dot_normal;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> surf_num_flux_minus_surf_soln_ext_dot_normal;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        //Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        //The way it is stored in metric_operators is to use sum-factorization in each direction,
        //but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        //Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        //This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        std::array<real,nstate> diss_soln_num_flux;
        std::array<real,nstate> soln_state_int;
        std::array<real,nstate> soln_state_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
        }
        diss_soln_num_flux = this->diss_num_flux_double->evaluate_solution_flux(soln_state_int, soln_state_ext, unit_phys_normal_int);

        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim].resize(n_face_quad_pts);
                    surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim].resize(n_face_quad_pts);
                }
                //solve
                surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q_int[istate][iquad]) * unit_phys_normal_int[idim] * face_Jac_norm_scaled;

                surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q_ext[istate][iquad]) * (- unit_phys_normal_int[idim]) * face_Jac_norm_scaled;
            }
        }
    }
    //solve residual and set
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();
    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            std::vector<real> rhs_int(n_shape_fns_int);

            soln_basis_int.inner_product_surface_1D(iface, 
                                                    surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim],
                                                    surf_quad_weights, rhs_int,
                                                    soln_basis_int.oneD_surf_operator,
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                local_auxiliary_RHS_int[istate*n_shape_fns_int + ishape][idim] += rhs_int[ishape]; 
            }
            std::vector<real> rhs_ext(n_shape_fns_ext);

            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim],
                                                    surf_quad_weights, rhs_ext,
                                                    soln_basis_ext.oneD_surf_operator,
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                local_auxiliary_RHS_ext[istate*n_shape_fns_ext + ishape][idim] += rhs_ext[ishape]; 
            }
        }
    }
}

/****************************************************
*
* PRIMARY EQUATIONS STRONG FORM
*
****************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_strong(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index                  current_cell_index,
    const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
    const unsigned int                                     poly_degree,
    OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis,
    OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis,
    OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper,
    OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
    dealii::Vector<real>                                   &local_rhs_int_cell)
{
    (void) current_cell_index;

    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate; 
    const unsigned int n_quad_pts_1D  = this->oneD_quadrature_collection[poly_degree].size();
    assert(n_quad_pts == pow(n_quad_pts_1D, dim));
    const std::vector<double> &vol_quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
    const std::vector<double> &oneD_vol_quad_weights = this->oneD_quadrature_collection[poly_degree].get_weights();

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    // Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    // We immediately separate them by state as to be able to use sum-factorization
    // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    // mult would sum the states at the quadrature point.
    std::array<std::vector<real>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff;
    const unsigned int p_min_filtered = this->poly_degree_max_large_scales + 1;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0){
            soln_coeff[istate].resize(n_shape_fns);
        }
        soln_coeff[istate][ishape] = this->solution(cell_dofs_indices[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = this->auxiliary_solution[idim](cell_dofs_indices[idof]);
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
    }
    std::array<std::vector<real>,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_q; //auxiliary sol at flux nodes
    std::vector<std::array<real,nstate>> soln_at_q_for_max_CFL(n_quad_pts);//Need soln written in a different for to use pre-existing max CFL function
    // Interpolate each state to the quadrature points using sum-factorization
    // with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        soln_at_q[istate].resize(n_quad_pts);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                         soln_basis.oneD_vol_operator);
        for(int idim=0; idim<dim; idim++){
            aux_soln_at_q[istate][idim].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_q[istate][idim],
                                             soln_basis.oneD_vol_operator);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            soln_at_q_for_max_CFL[iquad][istate] = soln_at_q[istate][iquad];
        }
    }

    // -- Solution at legendre poly
    std::array<std::vector<real>,nstate> legendre_soln_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_q; // legendre auxiliary sol at flux nodes
    if(this->do_compute_filtered_solution) {
        //==================================================
        // GET THE PRIMITIVE SOLUTION
        //==================================================
        std::array<std::vector<real>,nstate> primitive_soln_at_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_q; // primitive auxiliary sol at flux nodes
        // Resize the primitive soln arrays
        for(int istate=0; istate<nstate; istate++){
            primitive_soln_at_q[istate].resize(n_quad_pts);
            for(int idim=0; idim<dim; idim++){
                primitive_aux_soln_at_q[istate][idim].resize(n_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_q[istate][idim][iquad];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_q[istate][iquad] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_q[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                }
            }
        }
        
        //==================================================
        // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
        //==================================================
        // -- Primitive solution at legendre poly
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_q; // legendre auxiliary sol at flux nodes

        // Details: this projects to Legendre basis, truncates, then interpolates back to quad nodes.
        // -- Constructor for tensor product polynomials based on Polynomials::Legendre interpolation. 
        dealii::FE_DGQLegendre<1,1> legendre_poly_1D(poly_degree);
        // -- Projection operator for legendre basis
        OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper(1, poly_degree, this->max_grid_degree);
        legendre_soln_basis_projection_oper.build_1D_volume_operator(legendre_poly_1D, this->oneD_quadrature_collection[poly_degree]);
        // -- Legendre basis functions 
        OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis(1, poly_degree, this->max_grid_degree);
        legendre_soln_basis.build_1D_volume_operator(legendre_poly_1D, this->oneD_quadrature_collection[poly_degree]);
        for(int istate=0; istate<nstate; istate++){
            //==================================================
            // Solution
            //==================================================
            // -- (1) Project to Legendre basis
            std::vector<real> legendre_soln_coeff(n_shape_fns);
            legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_soln_at_q[istate], legendre_soln_coeff,
                                                                      legendre_soln_basis_projection_oper.oneD_vol_operator);
            // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
            if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                    if(ishape < p_min_filtered){
                        legendre_soln_coeff[ishape] = 0.0;
                    }
                }    
            }
            // -- (3) Interpolate filtered solution back to quadrature points
            primitive_legendre_soln_at_q[istate].resize(n_quad_pts);
            legendre_soln_basis.matrix_vector_mult_1D(legendre_soln_coeff, primitive_legendre_soln_at_q[istate],
                                                      legendre_soln_basis.oneD_vol_operator);
            //==================================================

            //==================================================
            // Auxiliary Solution (gradients)
            //==================================================
            dealii::Tensor<1,dim,std::vector<real>> legendre_aux_soln_coeff;
            for(int idim=0; idim<dim; idim++){
                // -- (1) Project to Legendre basis
                legendre_aux_soln_coeff[idim].resize(n_shape_fns);
                if(this->use_auxiliary_eq){
                    legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_aux_soln_at_q[istate][idim], legendre_aux_soln_coeff[idim],
                                                                              legendre_soln_basis_projection_oper.oneD_vol_operator);
                    // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                    if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                            if(ishape < p_min_filtered){
                                legendre_aux_soln_coeff[idim][ishape] = 0.0;
                            }
                        }    
                    }
                }
                else {
                    for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                        legendre_aux_soln_coeff[idim][ishape] = 0.0;
                    }
                }
                // -- (3) Interpolate filtered solution back to quadrature points
                primitive_legendre_aux_soln_at_q[istate][idim].resize(n_quad_pts);
                legendre_soln_basis.matrix_vector_mult_1D(legendre_aux_soln_coeff[idim], primitive_legendre_aux_soln_at_q[istate][idim],
                                                          legendre_soln_basis.oneD_vol_operator);
            }
            //==================================================
        }
        //=======================================================
        // CONVERT PRIMITIVE LEGENDRE SOLUTION TO CONSERVATIVE
        //=======================================================
        // Resize the conservative soln arrays
        for(int istate=0; istate<nstate; istate++){
            legendre_soln_at_q[istate].resize(n_quad_pts);
            for(int idim=0; idim<dim; idim++){
                legendre_aux_soln_at_q[istate][idim].resize(n_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_q[istate][idim][iquad];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_q[istate][iquad] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_q[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
    }

    // For pseudotime, we need to compute the time_scaled_solution.
    // Thus, we need to evaluate the max_dt_cell (as previously done in dg/weak_dg.cpp -> assemble_volume_term_explicit)
    // Get max artificial dissipation
    real max_artificial_diss = 0.0;
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        real artificial_diss_coeff_at_q = 0.0;
        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = this->volume_quadrature_collection[poly_degree].point(iquad);
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
            max_artificial_diss = std::max(artificial_diss_coeff_at_q, max_artificial_diss);
        }
    }
    // Get max_dt_cell for time_scaled_solution with pseudotime
    real cell_volume_estimate = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        cell_volume_estimate += metric_oper.det_Jac_vol[iquad] * vol_quad_weights[iquad];
    }
    const real cell_volume = cell_volume_estimate;
    const real diameter = cell->diameter();
    const real cell_diameter = cell_volume / std::pow(diameter,dim-1);
    const real cell_radius = 0.5 * cell_diameter;
    this->cell_volume[current_cell_index] = cell_volume;
    this->max_dt_cell[current_cell_index] = this->evaluate_CFL ( soln_at_q_for_max_CFL, max_artificial_diss, cell_radius, poly_degree);

    //get entropy projected variables
    std::array<std::vector<real>,nstate> entropy_var_at_q;
    std::array<std::vector<real>,nstate> projected_entropy_var_at_q;
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            entropy_var_at_q[istate].resize(n_quad_pts);
            projected_entropy_var_at_q[istate].resize(n_quad_pts);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<real,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<real,nstate> entropy_var;
            entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            std::vector<real> entropy_var_coeff(n_shape_fns);;
            soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_at_q[istate],
                                                             entropy_var_coeff,
                                                             soln_basis_projection_oper.oneD_vol_operator);
            soln_basis.matrix_vector_mult_1D(entropy_var_coeff,
                                             projected_entropy_var_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
    }


    //Compute the physical fluxes, then convert them into reference fluxes.
    //From the paper: Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.
    //For conservative DG, we compute the reference flux as per Eq. (9), to then recover the second volume integral in Eq. (17).
    //For curvilinear split-form in Eq. (22), we apply a two-pt flux of the metric-cofactor matrix on the matrix operator constructed by the entropy stable/conservtive 2pt flux.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_q;
    std::array<std::vector<real>,nstate> source_at_q;
    std::array<std::vector<real>,nstate> physical_source_at_q;

    // The matrix of two-pt fluxes for Hadamard products
    std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> conv_ref_2pt_flux_at_q;
    //Hadamard tensor-product sparsity pattern
    std::vector<std::array<unsigned int,dim>> Hadamard_rows_sparsity(n_quad_pts * n_quad_pts_1D);//size n^{d+1}
    std::vector<std::array<unsigned int,dim>> Hadamard_columns_sparsity(n_quad_pts * n_quad_pts_1D);
    //allocate reference 2pt flux for Hadamard product
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                conv_ref_2pt_flux_at_q[istate][idim].reinit(n_quad_pts, n_quad_pts_1D);//size n^d x n
            }
        }
        //extract the dof pairs that give non-zero entries for each direction
        //to use the "sum-factorized" Hadamard product.
        flux_basis.sum_factorized_Hadamard_sparsity_pattern(n_quad_pts_1D, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity);
    }


    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        //extract soln and auxiliary soln at quad pt to be used in physics
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        std::array<real,nstate> filtered_soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_q[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_state[istate] = legendre_soln_at_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_q[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state[istate][idim] = legendre_aux_soln_at_q[istate][idim][iquad];
            }
        }

        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        // Evaluate physical convective flux
        // If 2pt flux, transform to reference at construction to improve performance.
        // We technically use a REFERENCE 2pt flux for all entropy stable schemes.
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //get the soln for iquad from projected entropy variables
            std::array<real,nstate> entropy_var;
            for(int istate=0; istate<nstate; istate++){
                entropy_var[istate] = projected_entropy_var_at_q[istate][iquad];
            }
            soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
            
            //loop over all the non-zero entries for "sum-factorized" Hadamard product that corresponds to the iquad.
            for(unsigned int row_index = iquad * n_quad_pts_1D, column_index = 0; 
               // Hadamard_rows_sparsity[row_index][0] == iquad; 
                column_index < n_quad_pts_1D; 
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index][0] != iquad){
                    pcout<<"The volume Hadamard rows sparsity pattern does not match. Aborting..."<<std::endl;
                    std::abort();
                }

                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.

                for(int ref_dim=0; ref_dim<dim; ref_dim++){
                    const unsigned int flux_quad = Hadamard_columns_sparsity[row_index][ref_dim];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.

                    dealii::Tensor<2,dim,real> metric_cofactor_flux_basis;
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                            metric_cofactor_flux_basis[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][flux_quad];
                        }
                    }
                    std::array<real,nstate> soln_state_flux_basis;
                    std::array<real,nstate> entropy_var_flux_basis;
                    for(int istate=0; istate<nstate; istate++){
                        entropy_var_flux_basis[istate] = projected_entropy_var_at_q[istate][flux_quad];
                    }
                    soln_state_flux_basis = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_flux_basis);

                    //Compute the physical flux
                    std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                    conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_flux_basis);
                     
                    for(int istate=0; istate<nstate; istate++){
                        dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                        //For each state, transform the physical flux to a reference flux.
                        metric_oper.transform_physical_to_reference(
                            conv_phys_flux_2pt[istate],
                            0.5*(metric_cofactor + metric_cofactor_flux_basis),
                            conv_ref_flux_2pt);
                        //write into reference Hadamard flux matrix
                        conv_ref_2pt_flux_at_q[istate][ref_dim][iquad][column_index] = conv_ref_flux_2pt[ref_dim];
                    }
                }
            }
        }
        else{
            //Compute the physical flux
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        //Diffusion
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        //Compute the physical dissipative flux
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, filtered_soln_state, filtered_aux_soln_state, current_cell_index);

        // Manufactured source
        std::array<real,nstate> manufactured_source;
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim,real> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the manufactured source
            manufactured_source = this->pde_physics_double->source_term (vol_flux_node, soln_state, this->current_time, current_cell_index);
        }

        // Physical source
        std::array<real,nstate> physical_source;
        if(this->pde_physics_double->has_nonzero_physical_source) {
            dealii::Point<dim,real> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the physical source
            physical_source = this->pde_physics_double->physical_source_term (vol_flux_node, soln_state, aux_soln_state, current_cell_index);
        }

        //Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            //Trnasform to reference fluxes
            if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                //Do Nothing. 
                //I am leaving this block here so the diligent reader
                //remembers that, for entropy stable schemes, we construct
                //a REFERENCE two-point flux at construction, where the physical
                //to reference transformation was done by splitting the metric cofactor.
            }
            else{
                //transform the conservative convective physical flux to reference space
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor,
                    conv_ref_flux);
            }
            //transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor,
                diffusive_ref_flux);

            //Write the data in a way that we can use sum-factorization on.
            //Since sum-factorization improves the speed for matrix-vector multiplications,
            //We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                    diffusive_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                }
                //write data
                if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                    //Do nothing because written in a Hadamard product sum-factorized form above.
                }
                else{
                    conv_ref_flux_at_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                diffusive_ref_flux_at_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                if(iquad == 0){
                    source_at_q[istate].resize(n_quad_pts);
                }
                source_at_q[istate][iquad] = manufactured_source[istate];
            }
            if(this->pde_physics_double->has_nonzero_physical_source) {
                if(iquad == 0){
                    physical_source_at_q[istate].resize(n_quad_pts);
                }
                physical_source_at_q[istate][iquad] = physical_source[istate];
            }
        }
    }

    // Get a flux basis reference gradient operator in a sum-factorized Hadamard product sparse form. Then apply the divergence.
    std::array<dealii::FullMatrix<real>,dim> flux_basis_stiffness_skew_symm_oper_sparse;
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int idim=0; idim<dim; idim++){
            flux_basis_stiffness_skew_symm_oper_sparse[idim].reinit(n_quad_pts, n_quad_pts_1D);
        }
        flux_basis.sum_factorized_Hadamard_basis_assembly(n_quad_pts_1D, n_quad_pts_1D, 
                                                          Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                          flux_basis_stiffness.oneD_skew_symm_vol_oper, 
                                                          oneD_vol_quad_weights,
                                                          flux_basis_stiffness_skew_symm_oper_sparse);
    }

    //For each state we:
    //  1. Compute reference divergence.
    //  2. Then compute and write the rhs for the given state.
    for(int istate=0; istate<nstate; istate++){

        //Compute reference divergence of the reference fluxes.
        std::vector<real> conv_flux_divergence(n_quad_pts); 
        std::vector<real> diffusive_flux_divergence(n_quad_pts); 

        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //2pt flux Hadamard Product, and then multiply by vector of ones scaled by 1.
            // Same as the volume term in Eq. (15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485. but, 
            // where we use the reference skew-symmetric stiffness operator of the flux basis for the Q operator and the reference two-point flux as to make use of Alex's Hadamard product
            // sum-factorization type algorithm that exploits the structure of the flux basis in the reference space to have O(n^{d+1}).

            for(int ref_dim=0; ref_dim<dim; ref_dim++){
                dealii::FullMatrix<real> divergence_ref_flux_Hadamard_product(n_quad_pts, n_quad_pts_1D);
                flux_basis.Hadamard_product(flux_basis_stiffness_skew_symm_oper_sparse[ref_dim], conv_ref_2pt_flux_at_q[istate][ref_dim], divergence_ref_flux_Hadamard_product); 
                //Hadamard product times the vector of ones.
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    if(ref_dim == 0){
                        conv_flux_divergence[iquad] = 0.0;
                    }
                    for(unsigned int iquad_1D=0; iquad_1D<n_quad_pts_1D; iquad_1D++){
                        conv_flux_divergence[iquad] += divergence_ref_flux_Hadamard_product[iquad][iquad_1D];
                    }
                }
            }
            
        }
        else{
            //Reference divergence of the reference convective flux.
            flux_basis.divergence_matrix_vector_mult_1D(conv_ref_flux_at_q[istate], conv_flux_divergence,
                                                        flux_basis.oneD_vol_operator,
                                                        flux_basis.oneD_grad_operator);
        }
        //Reference divergence of the reference diffusive flux.
        flux_basis.divergence_matrix_vector_mult_1D(diffusive_ref_flux_at_q[istate], diffusive_flux_divergence,
                                                    flux_basis.oneD_vol_operator,
                                                    flux_basis.oneD_grad_operator);


        // Strong form
        // The right-hand side sends all the term to the side of the source term
        // Therefore, 
        // \divergence ( Fconv + Fdiss ) = source 
        // has the right-hand side
        // rhs = - \divergence( Fconv + Fdiss ) + source 
        // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
        // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
        std::vector<real> rhs(n_shape_fns);

        // Convective
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(conv_flux_divergence, ones, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }
        else {
            soln_basis.inner_product_1D(conv_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }

        // Diffusive
        // Note that for diffusion, the negative is defined in the physics. Since we used the auxiliary
        // variable, put a negative here.
        soln_basis.inner_product_1D(diffusive_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, true, -1.0);

        // Manufactured source
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            std::vector<real> JxW(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxW[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }
            soln_basis.inner_product_1D(source_at_q[istate], JxW, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        // Physical source
        if(this->pde_physics_double->has_nonzero_physical_source) {
            std::vector<real> JxW(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxW[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }
            soln_basis.inner_product_1D(physical_source_at_q[istate], JxW, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_int_cell(istate*n_shape_fns + ishape) += rhs[ishape];
        }

    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_strong(
    const unsigned int iface, 
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const unsigned int poly_degree, 
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices,
    OPERATOR::basis_functions<dim,2*dim,real> &soln_basis,
    OPERATOR::basis_functions<dim,2*dim,real> &flux_basis,
    OPERATOR::vol_projection_operator<dim,2*dim,real> &soln_basis_projection_oper,
    OPERATOR::metric_operators<real,dim,2*dim> &metric_oper,
    dealii::Vector<real> &local_rhs_cell)
{
    (void) current_cell_index;

    const unsigned int n_face_quad_pts  = this->face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol   = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_1D    = this->oneD_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs / nstate; 
    const std::vector<double> &face_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();

    AssertDimension (n_dofs, dof_indices.size());

    // Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    // We immediately separate them by state as to be able to use sum-factorization
    // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    // mult would sum the states at the quadrature point.
    std::array<std::vector<real>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff;
    const unsigned int p_min_filtered = this->poly_degree_max_large_scales + 1;
    for (unsigned int idof = 0; idof < n_dofs; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        // allocate
        if(ishape == 0){
            soln_coeff[istate].resize(n_shape_fns);
        }
        // solve
        soln_coeff[istate][ishape] = this->solution(dof_indices[idof]);
        for(int idim=0; idim<dim; idim++){
            //allocate
            if(ishape == 0){
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            }
            //solve
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = this->auxiliary_solution[idim](dof_indices[idof]);
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
    }
    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> soln_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_vol_q[istate].resize(n_quad_pts_vol);
        //solve soln at volume cubature nodes
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_vol_q[istate],
                                         soln_basis.oneD_vol_operator);

        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);

        for(int idim=0; idim<dim; idim++){
            //alocate
            aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
            //solve auxiliary soln at volume cubature nodes
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_vol_q[istate][idim],
                                             soln_basis.oneD_vol_operator);

            //allocate
            aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            //solve auxiliary soln at facet cubature nodes
            soln_basis.matrix_vector_mult_surface_1D(iface,
                                                     aux_soln_coeff[istate][idim], aux_soln_at_surf_q[istate][idim],
                                                     soln_basis.oneD_surf_operator,
                                                     soln_basis.oneD_vol_operator);
        }
    }

    // -- Solution at legendre poly
    // -- (a) Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> legendre_soln_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_vol_q; // legendre auxiliary sol at flux nodes
    // -- (b) Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> legendre_soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_surf_q; // legendre auxiliary sol at flux nodes
    if(this->do_compute_filtered_solution) {
        //==================================================
        // GET THE PRIMITIVE SOLUTION
        //==================================================
        std::array<std::vector<real>,nstate> primitive_soln_at_vol_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_vol_q;
        std::array<std::vector<real>,nstate> primitive_soln_at_surf_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_surf_q;
        // Resize the primitive soln arrays
        for(int istate=0; istate<nstate; istate++){
            primitive_soln_at_vol_q[istate].resize(n_quad_pts_vol);
            primitive_soln_at_surf_q[istate].resize(n_face_quad_pts);
            for(int idim=0; idim<dim; idim++){
                primitive_aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                primitive_aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        // -- volume
        for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_vol_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_vol_q[istate][idim][iquad];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_vol_q[istate][iquad] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_vol_q[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_surf_q[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_surf_q[istate][idim][iquad_face];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_surf_q[istate][iquad_face] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_surf_q[istate][idim][iquad_face] = primitive_aux_soln_state[istate][idim];
                }
            }
        }

        //==================================================
        // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
        //==================================================
        // -- Primitive solution at legendre poly
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_vol_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_vol_q;
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_surf_q;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_surf_q;

        // Details: this projects to Legendre basis, truncates, then interpolates back to quad nodes.
        // -- Constructor for tensor product polynomials based on Polynomials::Legendre interpolation. 
        dealii::FE_DGQLegendre<1,1> legendre_poly_1D(poly_degree);
        // -- Projection operator for legendre basis
        OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper(1, poly_degree, this->max_grid_degree);
        legendre_soln_basis_projection_oper.build_1D_volume_operator(legendre_poly_1D, this->oneD_quadrature_collection[poly_degree]);
        // -- Legendre basis functions 
        OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis(1, poly_degree, this->max_grid_degree);
        legendre_soln_basis.build_1D_volume_operator(legendre_poly_1D, this->oneD_quadrature_collection[poly_degree]);
        legendre_soln_basis.build_1D_surface_operator(legendre_poly_1D, this->oneD_face_quadrature);
        for(int istate=0; istate<nstate; istate++){
            //==================================================
            // Solution
            //==================================================
            // -- (1) Project to Legendre basis
            std::vector<real> legendre_soln_coeff(n_shape_fns);
            legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_soln_at_vol_q[istate], legendre_soln_coeff,
                                                                      legendre_soln_basis_projection_oper.oneD_vol_operator);
            // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
            if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                    if(ishape < p_min_filtered){
                        legendre_soln_coeff[ishape] = 0.0;
                    }
                }    
            }
            // -- (3) Interpolate filtered solution back to quadrature points
            primitive_legendre_soln_at_vol_q[istate].resize(n_quad_pts_vol);
            legendre_soln_basis.matrix_vector_mult_1D(legendre_soln_coeff, primitive_legendre_soln_at_vol_q[istate],
                                                      legendre_soln_basis.oneD_vol_operator);
            primitive_legendre_soln_at_surf_q[istate].resize(n_face_quad_pts);
            legendre_soln_basis.matrix_vector_mult_surface_1D(iface,
                                                              legendre_soln_coeff, primitive_legendre_soln_at_surf_q[istate],
                                                              legendre_soln_basis.oneD_surf_operator,
                                                              legendre_soln_basis.oneD_vol_operator);
            //==================================================

            //==================================================
            // Auxiliary Solution (gradients)
            //==================================================
            dealii::Tensor<1,dim,std::vector<real>> legendre_aux_soln_coeff;
            for(int idim=0; idim<dim; idim++){
                // -- (1) Project to Legendre basis
                legendre_aux_soln_coeff[idim].resize(n_shape_fns);
                if(this->use_auxiliary_eq){
                    legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_aux_soln_at_vol_q[istate][idim], legendre_aux_soln_coeff[idim],
                                                                              legendre_soln_basis_projection_oper.oneD_vol_operator);
                    // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                    if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                            if(ishape < p_min_filtered){
                                legendre_aux_soln_coeff[idim][ishape] = 0.0;
                            }
                        }
                    }
                }
                else {
                    for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                        legendre_aux_soln_coeff[idim][ishape] = 0.0;
                    }
                }
                // -- (3) Interpolate filtered solution back to quadrature points
                primitive_legendre_aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                legendre_soln_basis.matrix_vector_mult_1D(legendre_aux_soln_coeff[idim], primitive_legendre_aux_soln_at_vol_q[istate][idim],
                                                          legendre_soln_basis.oneD_vol_operator);
                primitive_legendre_aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
                legendre_soln_basis.matrix_vector_mult_surface_1D(iface,
                                                                  legendre_aux_soln_coeff[idim], primitive_legendre_aux_soln_at_surf_q[istate][idim],
                                                                  legendre_soln_basis.oneD_surf_operator,
                                                                  legendre_soln_basis.oneD_vol_operator);
            }
            //==================================================
        }
        //=======================================================
        // CONVERT PRIMITIVE LEGENDRE SOLUTION TO CONSERVATIVE
        //=======================================================
        // Resize the conservative soln arrays
        for(int istate=0; istate<nstate; istate++){
            legendre_soln_at_vol_q[istate].resize(n_quad_pts_vol);
            legendre_soln_at_surf_q[istate].resize(n_face_quad_pts);
            for(int idim=0; idim<dim; idim++){
                legendre_aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                legendre_aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        // -- volume
        for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_vol_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_vol_q[istate][idim][iquad];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_vol_q[istate][iquad] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_vol_q[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface
        for (unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_surf_q[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_surf_q[istate][idim][iquad_face];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_surf_q[istate][iquad_face] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_surf_q[istate][idim][iquad_face] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
    }

    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor_vol;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        std::array<real,nstate> filtered_soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_state[istate] = legendre_soln_at_vol_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state[istate][idim] = legendre_aux_soln_at_vol_q[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, filtered_soln_state, filtered_aux_soln_state, current_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                    diffusive_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                }
                //write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                diffusive_ref_flux_at_vol_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const int dim_not_zero = iface / 2;//reference direction of face integer division

    std::array<std::vector<real>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        //solve
        //Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet

        //interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis.matrix_vector_mult_surface_1D(iface, 
                                                     conv_ref_flux_at_vol_q[istate][dim_not_zero],
                                                     conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                     flux_basis.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero]);//don't add to previous value, scale by unit_normal int
        }

        //interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis.matrix_vector_mult_surface_1D(iface, 
                                                 diffusive_ref_flux_at_vol_q[istate][dim_not_zero],
                                                 diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                 flux_basis.oneD_surf_operator,
                                                 flux_basis.oneD_vol_operator,
                                                 false, unit_ref_normal_int[dim_not_zero]);
    }

    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<real>,nstate> entropy_var_vol;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol[istate].resize(n_quad_pts_vol);
            }
            entropy_var_vol[istate][iquad] = entropy_var[istate];
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<real>,nstate> projected_entropy_var_vol;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol[istate].resize(n_quad_pts_vol);
        projected_entropy_var_surf[istate].resize(n_face_quad_pts);

        //interior
        std::vector<real> entropy_var_coeff(n_shape_fns);
        soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_vol[istate],
                                                         entropy_var_coeff,
                                                         soln_basis_projection_oper.oneD_vol_operator);
        soln_basis.matrix_vector_mult_1D(entropy_var_coeff,
                                         projected_entropy_var_vol[istate],
                                         soln_basis.oneD_vol_operator);
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 entropy_var_coeff, 
                                                 projected_entropy_var_surf[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size = n_face_quad_pts * n_quad_pts_1D;
    const unsigned int col_size = n_face_quad_pts * n_quad_pts_1D;
    std::vector<unsigned int> Hadamard_rows_sparsity(row_size);
    std::vector<unsigned int> Hadamard_columns_sparsity(col_size);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity, dim_not_zero);
    }

    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux[istate].reinit(n_face_quad_pts, n_quad_pts_1D);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,real> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }
             
            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<real,nstate> entropy_var_face;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face[istate] = projected_entropy_var_surf[istate][iquad_face];
            }
            std::array<real,nstate> soln_state_face;
            soln_state_face= this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face);

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D, column_index = 0; 
                column_index < n_quad_pts_1D;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index] != iquad_face){
                    pcout<<"The boundary Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero];
                }
            }
        }
        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol= this->oneD_quadrature_collection[poly_degree].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse(n_face_quad_pts, n_quad_pts_1D);
        flux_basis.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D, 
                                                                  Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                                  flux_basis.oneD_surf_operator[iface_1D], 
                                                                  oneD_quad_weights_vol,
                                                                  surf_oper_sparse,
                                                                  dim_not_zero);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            dealii::FullMatrix<real> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D);
            flux_basis.Hadamard_product(surf_oper_sparse, 
                                        surface_ref_2pt_flux[istate], 
                                        surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol[istate].resize(n_quad_pts_vol);
         
            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf[istate][iface_quad] 
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol[istate][Hadamard_columns_sparsity[column_index]] 
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                }
            }
        }
    }//end of if split form or curvilinear split form


    //the outward reference normal dircetion.
    std::array<std::vector<real>,nstate> conv_flux_dot_normal;
    std::array<std::vector<real>,nstate> diss_flux_dot_normal_diff;
    // Get surface numerical fluxes
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper.transform_reference_to_physical(unit_ref_normal_int,
                                                    metric_cofactor_surf,
                                                    unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        //get the projected entropy variables, soln, and 
        //auxiliary solution on the surface point.
        std::array<real,nstate> entropy_var_face_int;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_int;
        std::array<real,nstate> soln_interp_to_face_int;
        std::array<real,nstate> filtered_soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_state[istate] = legendre_soln_at_surf_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state[istate][idim] = legendre_aux_soln_at_surf_q[istate][idim][iquad];
            }
        }

        //extract solution on surface from projected entropy variables
        std::array<real,nstate> soln_state_int;
        soln_state_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);

        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q[istate][iquad];
            }
        }

        std::array<real,nstate> soln_boundary;
        std::array<dealii::Tensor<1,dim,real>,nstate> grad_soln_boundary;
        dealii::Point<dim,real> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
        }
        //I am not sure if BC should be from solution interpolated to face
        //or solution from the projected entropy variables.
        //Now, it uses projected entropy variables for NSFR, and solution
        //interpolated to face for conservative DG.
        this->pde_physics_double->boundary_face_values (boundary_id, surf_flux_node, unit_phys_normal_int, soln_state_int, aux_soln_state_int, filtered_soln_state, filtered_aux_soln_state, soln_boundary, grad_soln_boundary);
        
        // Convective numerical flux.
        std::array<real,nstate> conv_num_flux_dot_n_at_q;
        conv_num_flux_dot_n_at_q = this->conv_num_flux_double->evaluate_flux(soln_state_int, soln_boundary, unit_phys_normal_int);
        
        // Dissipative numerical flux
        std::array<real,nstate> diss_auxi_num_flux_dot_n_at_q;
        diss_auxi_num_flux_dot_n_at_q = this->diss_num_flux_double->evaluate_auxiliary_flux(
            current_cell_index, current_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_boundary,
            aux_soln_state_int, grad_soln_boundary,
            filtered_soln_state, soln_boundary,
            filtered_aux_soln_state, grad_soln_boundary,
            unit_phys_normal_int, penalty, true, boundary_id);

        for(int istate=0; istate<nstate; istate++){
            // allocate
            if(iquad==0){
                conv_flux_dot_normal[istate].resize(n_face_quad_pts);
                diss_flux_dot_normal_diff[istate].resize(n_face_quad_pts);
            }
            // write data
            conv_flux_dot_normal[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
            diss_flux_dot_normal_diff[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate]
                                                     - diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate][iquad];
        }
    }

    //solve rhs
    for(int istate=0; istate<nstate; istate++){
        std::vector<real> rhs(n_shape_fns);
        //Convective flux on the facet
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis.inner_product_surface_1D(iface, 
                                                surf_vol_ref_2pt_flux_interp_surf[istate], 
                                                ones_surf, rhs, 
                                                soln_basis.oneD_surf_operator, 
                                                soln_basis.oneD_vol_operator,
                                                false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol, 1.0);
            soln_basis.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol[istate], 
                                            ones_vol, rhs, 
                                            soln_basis.oneD_vol_operator, 
                                            true, -1.0);
        }
        else{
            soln_basis.inner_product_surface_1D(iface, conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                face_quad_weights, rhs, 
                                                soln_basis.oneD_surf_operator, 
                                                soln_basis.oneD_vol_operator,
                                                false, 1.0);//adding=false, scaled by factor=-1.0 bc subtract it
        }
        //Convective surface nnumerical flux.
        soln_basis.inner_product_surface_1D(iface, conv_flux_dot_normal[istate], 
                                            face_quad_weights, rhs, 
                                            soln_basis.oneD_surf_operator, 
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        //Dissipative surface numerical flux.
        soln_basis.inner_product_surface_1D(iface, diss_flux_dot_normal_diff[istate], 
                                            face_quad_weights, rhs, 
                                            soln_basis.oneD_surf_operator, 
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_cell(istate*n_shape_fns + ishape) += rhs[ishape];
        }
    }
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_strong(
    const unsigned int iface, const unsigned int neighbor_iface, 
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int poly_degree_int, 
    const unsigned int poly_degree_ext, 
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    OPERATOR::basis_functions<dim,2*dim,real>          &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>          &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim,real>          &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim,real>          &flux_basis_ext,
    OPERATOR::vol_projection_operator<dim,2*dim,real>  &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim,real>  &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_int,
    OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_ext,
    dealii::Vector<real>                               &local_rhs_int_cell,
    dealii::Vector<real>                               &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work

    const unsigned int n_quad_pts_vol_int  = this->volume_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_vol_ext  = this->volume_quadrature_collection[poly_degree_ext].size();
    const unsigned int n_quad_pts_1D_int  = this->oneD_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_1D_ext  = this->oneD_quadrature_collection[poly_degree_ext].size();

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Extract interior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff_int;
    const unsigned int p_min_filtered = this->poly_degree_max_large_scales + 1;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_int].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_int].system_to_component_index(idof).second;
        if(ishape == 0){
            soln_coeff_int[istate].resize(n_shape_fns_int);
        }
        soln_coeff_int[istate][ishape] = this->solution(dof_indices_int[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_int[istate][idim].resize(n_shape_fns_int);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_int[istate][idim][ishape] = this->auxiliary_solution[idim](dof_indices_int[idof]);
            }
            else{
                aux_soln_coeff_int[istate][idim][ishape] = 0.0;
            }
        }
    }

    // Extract exterior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff_ext;
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_ext].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_ext].system_to_component_index(idof).second;
        if(ishape == 0){
            soln_coeff_ext[istate].resize(n_shape_fns_ext);
        }
        soln_coeff_ext[istate][ishape] = this->solution(dof_indices_ext[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_ext[istate][idim].resize(n_shape_fns_ext);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_ext[istate][idim][ishape] = this->auxiliary_solution[idim](dof_indices_ext[idof]);
            }
            else{
                aux_soln_coeff_ext[istate][idim][ishape] = 0.0;
            }
        }
    }

    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> soln_at_vol_q_int;
    std::array<std::vector<real>,nstate> soln_at_vol_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q_ext;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> soln_at_surf_q_int;
    std::array<std::vector<real>,nstate> soln_at_surf_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        // allocate
        soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
        soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
        // solve soln at volume cubature nodes
        soln_basis_int.matrix_vector_mult_1D(soln_coeff_int[istate], soln_at_vol_q_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_1D(soln_coeff_ext[istate], soln_at_vol_q_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);

        // allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        // solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);

        for(int idim=0; idim<dim; idim++){
            // alocate
            aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
            aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
            // solve auxiliary soln at volume cubature nodes
            soln_basis_int.matrix_vector_mult_1D(aux_soln_coeff_int[istate][idim], aux_soln_at_vol_q_int[istate][idim],
                                                 soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_1D(aux_soln_coeff_ext[istate][idim], aux_soln_at_vol_q_ext[istate][idim],
                                                 soln_basis_ext.oneD_vol_operator);

            // allocate
            aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
            aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
            // solve auxiliary soln at facet cubature nodes
            soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                         aux_soln_coeff_int[istate][idim], aux_soln_at_surf_q_int[istate][idim],
                                                         soln_basis_int.oneD_surf_operator,
                                                         soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                         aux_soln_coeff_ext[istate][idim], aux_soln_at_surf_q_ext[istate][idim],
                                                         soln_basis_ext.oneD_surf_operator,
                                                         soln_basis_ext.oneD_vol_operator);
        }
    }

    // -- Solution at legendre poly
    // -- (a) Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> legendre_soln_at_vol_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_vol_q_int; // legendre auxiliary sol at flux nodes
    std::array<std::vector<real>,nstate> legendre_soln_at_vol_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_vol_q_ext; // legendre auxiliary sol at flux nodes
    // -- (b) Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> legendre_soln_at_surf_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_surf_q_int; // legendre auxiliary sol at flux nodes
    std::array<std::vector<real>,nstate> legendre_soln_at_surf_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_surf_q_ext; // legendre auxiliary sol at flux nodes
    if(this->do_compute_filtered_solution) {
        //==================================================
        // GET THE PRIMITIVE SOLUTION
        //==================================================
        std::array<std::vector<real>,nstate> primitive_soln_at_vol_q_int;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_vol_q_int;
        std::array<std::vector<real>,nstate> primitive_soln_at_surf_q_int;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_surf_q_int;
        std::array<std::vector<real>,nstate> primitive_soln_at_vol_q_ext;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_vol_q_ext;
        std::array<std::vector<real>,nstate> primitive_soln_at_surf_q_ext;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_surf_q_ext;
        // Resize the primitive soln arrays
        for(int istate=0; istate<nstate; istate++){
            primitive_soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
            primitive_soln_at_surf_q_int[istate].resize(n_face_quad_pts);
            primitive_soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
            primitive_soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
            for(int idim=0; idim<dim; idim++){
                primitive_aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                primitive_aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
                primitive_aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                primitive_aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        // -- volume int
        for (unsigned int iquad=0; iquad<n_quad_pts_vol_int; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_vol_q_int[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_vol_q_int[istate][idim][iquad];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_vol_q_int[istate][iquad] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_vol_q_int[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                }
            }
        }
        // -- volume ext
        for (unsigned int iquad=0; iquad<n_quad_pts_vol_ext; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_vol_q_ext[istate][idim][iquad];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_vol_q_ext[istate][iquad] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_vol_q_ext[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface int
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_surf_q_int[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_surf_q_int[istate][idim][iquad_face];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_surf_q_int[istate][iquad_face] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_surf_q_int[istate][idim][iquad_face] = primitive_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface ext
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            // extract conservative soln state
            std::array<real,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_surf_q_ext[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    aux_soln_state[istate][idim] = aux_soln_at_surf_q_ext[istate][idim][iquad_face];
                }
            }
            // compute primitive soln state from conservative
            std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
            // store primitive soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_surf_q_ext[istate][iquad_face] = primitive_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_surf_q_ext[istate][idim][iquad_face] = primitive_aux_soln_state[istate][idim];
                }
            }
        }

        //==================================================
        // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
        //==================================================
        // -- Primitive solution at legendre poly
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_vol_q_int;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_vol_q_int;
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_surf_q_int;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_surf_q_int;
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_vol_q_ext;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_vol_q_ext;
        std::array<std::vector<real>,nstate> primitive_legendre_soln_at_surf_q_ext;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_surf_q_ext;

        // Details: this projects to Legendre basis, truncates, then interpolates back to quad nodes.
        // -- Constructor for tensor product polynomials based on Polynomials::Legendre interpolation. 
        dealii::FE_DGQLegendre<1,1> legendre_poly_1D_int(poly_degree_int);
        dealii::FE_DGQLegendre<1,1> legendre_poly_1D_ext(poly_degree_ext);
        // -- Projection operator for legendre basis
        OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper_int(1, poly_degree_int, this->max_grid_degree);
        legendre_soln_basis_projection_oper_int.build_1D_volume_operator(legendre_poly_1D_int, this->oneD_quadrature_collection[poly_degree_int]);
        OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper_ext(1, poly_degree_ext, this->max_grid_degree);
        legendre_soln_basis_projection_oper_ext.build_1D_volume_operator(legendre_poly_1D_ext, this->oneD_quadrature_collection[poly_degree_ext]);
        // -- Legendre basis functions 
        OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis_int(1, poly_degree_int, this->max_grid_degree);
        legendre_soln_basis_int.build_1D_volume_operator(legendre_poly_1D_int, this->oneD_quadrature_collection[poly_degree_int]);
        legendre_soln_basis_int.build_1D_surface_operator(legendre_poly_1D_int, this->oneD_face_quadrature);
        OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis_ext(1, poly_degree_ext, this->max_grid_degree);
        legendre_soln_basis_ext.build_1D_volume_operator(legendre_poly_1D_ext, this->oneD_quadrature_collection[poly_degree_ext]);
        legendre_soln_basis_ext.build_1D_surface_operator(legendre_poly_1D_ext, this->oneD_face_quadrature);
        for(int istate=0; istate<nstate; istate++){
            //==================================================
            // Solution
            //==================================================
            // -- (1) Project to Legendre basis
            std::vector<real> legendre_soln_coeff_int(n_shape_fns_int);
            legendre_soln_basis_projection_oper_int.matrix_vector_mult_1D(primitive_soln_at_vol_q_int[istate], legendre_soln_coeff_int,
                                                                          legendre_soln_basis_projection_oper_int.oneD_vol_operator);
            std::vector<real> legendre_soln_coeff_ext(n_shape_fns_ext);
            legendre_soln_basis_projection_oper_ext.matrix_vector_mult_1D(primitive_soln_at_vol_q_ext[istate], legendre_soln_coeff_ext,
                                                                          legendre_soln_basis_projection_oper_ext.oneD_vol_operator);
            // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
            if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                    if(ishape < p_min_filtered){
                        legendre_soln_coeff_int[ishape] = 0.0;
                    }
                }
                for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                    if(ishape < p_min_filtered){
                        legendre_soln_coeff_ext[ishape] = 0.0;
                    }
                }
            }
            // -- (3) Interpolate filtered solution back to quadrature points
            primitive_legendre_soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
            legendre_soln_basis_int.matrix_vector_mult_1D(legendre_soln_coeff_int, primitive_legendre_soln_at_vol_q_int[istate],
                                                          legendre_soln_basis_int.oneD_vol_operator);
            primitive_legendre_soln_at_surf_q_int[istate].resize(n_face_quad_pts);
            legendre_soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                                  legendre_soln_coeff_int, primitive_legendre_soln_at_surf_q_int[istate],
                                                                  legendre_soln_basis_int.oneD_surf_operator,
                                                                  legendre_soln_basis_int.oneD_vol_operator);
            primitive_legendre_soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
            legendre_soln_basis_ext.matrix_vector_mult_1D(legendre_soln_coeff_ext, primitive_legendre_soln_at_vol_q_ext[istate],
                                                          legendre_soln_basis_ext.oneD_vol_operator);
            primitive_legendre_soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
            legendre_soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                                  legendre_soln_coeff_ext, primitive_legendre_soln_at_surf_q_ext[istate],
                                                                  legendre_soln_basis_ext.oneD_surf_operator,
                                                                  legendre_soln_basis_ext.oneD_vol_operator);
            //==================================================

            //==================================================
            // Auxiliary Solution (gradients)
            //==================================================
            dealii::Tensor<1,dim,std::vector<real>> legendre_aux_soln_coeff_int;
            dealii::Tensor<1,dim,std::vector<real>> legendre_aux_soln_coeff_ext;
            for(int idim=0; idim<dim; idim++){
                // -- (1) Project to Legendre basis
                legendre_aux_soln_coeff_int[idim].resize(n_shape_fns_int);
                legendre_aux_soln_coeff_ext[idim].resize(n_shape_fns_ext);
                if(this->use_auxiliary_eq){
                    legendre_soln_basis_projection_oper_int.matrix_vector_mult_1D(primitive_aux_soln_at_vol_q_int[istate][idim], legendre_aux_soln_coeff_int[idim],
                                                                                  legendre_soln_basis_projection_oper_int.oneD_vol_operator);
                    legendre_soln_basis_projection_oper_ext.matrix_vector_mult_1D(primitive_aux_soln_at_vol_q_ext[istate][idim], legendre_aux_soln_coeff_ext[idim],
                                                                                  legendre_soln_basis_projection_oper_ext.oneD_vol_operator);
                    // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                    if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                        for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                            if(ishape < p_min_filtered){
                                legendre_aux_soln_coeff_int[idim][ishape] = 0.0;
                            }
                        }
                        for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                            if(ishape < p_min_filtered){
                                legendre_aux_soln_coeff_ext[idim][ishape] = 0.0;
                            }
                        }
                    }
                }
                else {
                    for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                        legendre_aux_soln_coeff_int[idim][ishape] = 0.0;
                    }
                    for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                        legendre_aux_soln_coeff_ext[idim][ishape] = 0.0;
                    }
                }
                // -- (3) Interpolate filtered solution back to quadrature points
                primitive_legendre_aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                legendre_soln_basis_int.matrix_vector_mult_1D(legendre_aux_soln_coeff_int[idim], primitive_legendre_aux_soln_at_vol_q_int[istate][idim],
                                                              legendre_soln_basis_int.oneD_vol_operator);
                primitive_legendre_aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
                legendre_soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                                      legendre_aux_soln_coeff_int[idim], primitive_legendre_aux_soln_at_surf_q_int[istate][idim],
                                                                      legendre_soln_basis_int.oneD_surf_operator,
                                                                      legendre_soln_basis_int.oneD_vol_operator);
                primitive_legendre_aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                legendre_soln_basis_ext.matrix_vector_mult_1D(legendre_aux_soln_coeff_ext[idim], primitive_legendre_aux_soln_at_vol_q_ext[istate][idim],
                                                              legendre_soln_basis_ext.oneD_vol_operator);
                primitive_legendre_aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
                legendre_soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                                      legendre_aux_soln_coeff_ext[idim], primitive_legendre_aux_soln_at_surf_q_ext[istate][idim],
                                                                      legendre_soln_basis_ext.oneD_surf_operator,
                                                                      legendre_soln_basis_ext.oneD_vol_operator);
            }
            //==================================================
        }
        //=======================================================
        // CONVERT PRIMITIVE LEGENDRE SOLUTION TO CONSERVATIVE
        //=======================================================
        // Resize the conservative soln arrays
        for(int istate=0; istate<nstate; istate++){
            legendre_soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
            legendre_soln_at_surf_q_int[istate].resize(n_face_quad_pts);
            legendre_soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
            legendre_soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
            for(int idim=0; idim<dim; idim++){
                legendre_aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                legendre_aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
                legendre_aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                legendre_aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
            }
        }
        // Compute the primitive soln at all iquad and fill arrays
        // -- volume int
        for (unsigned int iquad=0; iquad<n_quad_pts_vol_int; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_vol_q_int[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_vol_q_int[istate][idim][iquad];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_vol_q_int[istate][iquad] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_vol_q_int[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
        // -- volume ext
        for (unsigned int iquad=0; iquad<n_quad_pts_vol_ext; ++iquad) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_vol_q_ext[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_vol_q_ext[istate][idim][iquad];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_vol_q_ext[istate][iquad] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_vol_q_ext[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface int
        for (unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_surf_q_int[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_surf_q_int[istate][idim][iquad_face];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_surf_q_int[istate][iquad_face] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_surf_q_int[istate][idim][iquad_face] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
        // -- surface ext
        for (unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++) {
            // extract conservative soln state
            std::array<real,nstate> primitive_legendre_soln_state;
            std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
            for(int istate=0; istate<nstate; istate++){
                primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_surf_q_ext[istate][iquad_face];
                for(int idim=0; idim<dim; idim++){
                    primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_surf_q_ext[istate][idim][iquad_face];
                }
            }
            // compute conservative soln state from primitive
            std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
            std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
            // store conservative soln at quadrature point
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_surf_q_ext[istate][iquad_face] = legendre_soln_state[istate];
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_surf_q_ext[istate][idim][iquad_face] = legendre_aux_soln_state[istate][idim];
                }
            }
        }
    }




    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q_int;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_int; ++iquad) {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor_vol_int;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        std::array<real,nstate> filtered_soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_state[istate] = legendre_soln_at_vol_q_int[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_int[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state[istate][idim] = legendre_aux_soln_at_vol_q_int[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        //Only for conservtive DG do we interpolate volume fluxes to the facet
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, filtered_soln_state, filtered_aux_soln_state, current_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_int.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_int,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper_int.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_int,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                    diffusive_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_int[istate][idim][iquad] = conv_ref_flux[idim];
                }
                diffusive_ref_flux_at_vol_q_int[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Next we do exterior volume reference fluxes.
    // Note we split the quad integrals because the interior and exterior could be of different poly basis
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q_ext;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_ext; ++iquad) {

        // Extract exterior volume metric cofactor matrix at given volume cubature node.
        dealii::Tensor<2,dim,real> metric_cofactor_vol_ext;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        std::array<real,nstate> filtered_soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_state[istate] = legendre_soln_at_vol_q_ext[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_ext[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state[istate][idim] = legendre_aux_soln_at_vol_q_ext[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, filtered_soln_state, filtered_aux_soln_state, neighbor_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_ext.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_ext,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper_ext.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_ext,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                    diffusive_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_ext[istate][idim][iquad] = conv_ref_flux[idim];
                }
                diffusive_ref_flux_at_vol_q_ext[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const dealii::Tensor<1,dim,double> unit_ref_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[neighbor_iface];
    // Extract the reference direction that is outward facing on the facet.
    const int dim_not_zero_int = iface / 2;//reference direction of face integer division
    const int dim_not_zero_ext = neighbor_iface / 2;//reference direction of face integer division

    std::array<std::vector<real>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        // solve
        // Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet
        
        // interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis_int.matrix_vector_mult_surface_1D(iface, 
                                                         conv_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                         conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_int.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                         flux_basis_int.oneD_vol_operator,
                                                         false, unit_ref_normal_int[dim_not_zero_int]);//don't add to previous value, scale by unit_normal int
            flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface, 
                                                         conv_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                         conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_ext.oneD_surf_operator,
                                                         flux_basis_ext.oneD_vol_operator,
                                                         false, unit_ref_normal_ext[dim_not_zero_ext]);//don't add to previous value, unit_normal ext is -unit normal int
        }

        // interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis_int.matrix_vector_mult_surface_1D(iface, 
                                                     diffusive_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                     diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_int.oneD_surf_operator,
                                                     flux_basis_int.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero_int]);
        flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface, 
                                                     diffusive_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                     diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_ext.oneD_surf_operator,
                                                     flux_basis_ext.oneD_vol_operator,
                                                     false, unit_ref_normal_ext[dim_not_zero_ext]);
    }


    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<real>,nstate> entropy_var_vol_int;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_int; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
            }
            entropy_var_vol_int[istate][iquad] = entropy_var[istate];
        }
    }
    std::array<std::vector<real>,nstate> entropy_var_vol_ext;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_ext; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
            }
            entropy_var_vol_ext[istate][iquad] = entropy_var[istate];
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<real>,nstate> projected_entropy_var_vol_int;
    std::array<std::vector<real>,nstate> projected_entropy_var_vol_ext;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf_int;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf_ext;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
        projected_entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
        projected_entropy_var_surf_int[istate].resize(n_face_quad_pts);
        projected_entropy_var_surf_ext[istate].resize(n_face_quad_pts);

        //interior
        std::vector<real> entropy_var_coeff_int(n_shape_fns_int);
        soln_basis_projection_oper_int.matrix_vector_mult_1D(entropy_var_vol_int[istate],
                                                             entropy_var_coeff_int,
                                                             soln_basis_projection_oper_int.oneD_vol_operator);
        soln_basis_int.matrix_vector_mult_1D(entropy_var_coeff_int,
                                             projected_entropy_var_vol_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     entropy_var_coeff_int, 
                                                     projected_entropy_var_surf_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);

        //exterior
        std::vector<real> entropy_var_coeff_ext(n_shape_fns_ext);
        soln_basis_projection_oper_ext.matrix_vector_mult_1D(entropy_var_vol_ext[istate],
                                                             entropy_var_coeff_ext,
                                                             soln_basis_projection_oper_ext.oneD_vol_operator);

        soln_basis_ext.matrix_vector_mult_1D(entropy_var_coeff_ext,
                                             projected_entropy_var_vol_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     entropy_var_coeff_ext, 
                                                     projected_entropy_var_surf_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    const unsigned int col_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    std::vector<unsigned int> Hadamard_rows_sparsity_int(row_size_int);
    std::vector<unsigned int> Hadamard_columns_sparsity_int(col_size_int);
    const unsigned int row_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    const unsigned int col_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    std::vector<unsigned int> Hadamard_rows_sparsity_ext(row_size_ext);
    std::vector<unsigned int> Hadamard_columns_sparsity_ext(col_size_ext);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis_int.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_int, Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int, dim_not_zero_int);
        flux_basis_ext.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_ext, Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext, dim_not_zero_ext);
    }

    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf_int;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf_ext;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol_int;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol_ext;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux_int;
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux_ext;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux_int[istate].reinit(n_face_quad_pts, n_quad_pts_1D_int);
            surface_ref_2pt_flux_ext[istate].reinit(n_face_quad_pts, n_quad_pts_1D_ext);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,real> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }
             
            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<real,nstate> entropy_var_face_int;
            std::array<real,nstate> entropy_var_face_ext;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad_face];
                entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad_face];
            }
            std::array<real,nstate> soln_state_face_int;
            soln_state_face_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
            std::array<real,nstate> soln_state_face_ext;
            soln_state_face_ext = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_int, column_index = 0; 
                column_index < n_quad_pts_1D_int;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_int[row_index] != iquad_face){
                    pcout<<"The interior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_int[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol_int;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_int[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face_int);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_int.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol_int),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_int[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero_int];
                }
            }
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_ext, column_index = 0; 
                column_index < n_quad_pts_1D_ext;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_ext[row_index] != iquad_face){
                    pcout<<"The exterior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_ext[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol_ext;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_ext[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face_ext);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_ext.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol_ext),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_ext[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero_ext];
                }
            }
        }

        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol_int = this->oneD_quadrature_collection[poly_degree_int].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_int(n_face_quad_pts, n_quad_pts_1D_int);
        flux_basis_int.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_int, 
                                                                      Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int,
                                                                      flux_basis_int.oneD_surf_operator[iface_1D], 
                                                                      oneD_quad_weights_vol_int,
                                                                      surf_oper_sparse_int,
                                                                      dim_not_zero_int);
        const int neighbor_iface_1D = neighbor_iface % 2;//the reference neighbour face number
        const std::vector<double> &oneD_quad_weights_vol_ext = this->oneD_quadrature_collection[poly_degree_ext].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_ext(n_face_quad_pts, n_quad_pts_1D_ext);
        flux_basis_ext.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_ext, 
                                                                      Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext,
                                                                      flux_basis_ext.oneD_surf_operator[neighbor_iface_1D], 
                                                                      oneD_quad_weights_vol_ext,
                                                                      surf_oper_sparse_ext,
                                                                      dim_not_zero_ext);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            dealii::FullMatrix<real> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D_int);
            flux_basis_int.Hadamard_product(surf_oper_sparse_int, 
                                            surface_ref_2pt_flux_int[istate], 
                                            surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            dealii::FullMatrix<real> surface_ref_2pt_flux_ext_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D_ext);
            flux_basis_ext.Hadamard_product(surf_oper_sparse_ext, 
                                            surface_ref_2pt_flux_ext[istate], 
                                            surface_ref_2pt_flux_ext_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf_int[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_surf_ext[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol_int[istate].resize(n_quad_pts_vol_int);
            surf_vol_ref_2pt_flux_interp_vol_ext[istate].resize(n_quad_pts_vol_ext);
         
            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D_int; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf_int[istate][iface_quad] 
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_int + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol_int[istate][Hadamard_columns_sparsity_int[column_index]] 
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                }
                for(unsigned int iquad_ext=0; iquad_ext<n_quad_pts_1D_ext; iquad_ext++){
                    surf_vol_ref_2pt_flux_interp_surf_ext[istate][iface_quad] 
                        -= surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad][iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_ext + iquad_ext;
                    surf_vol_ref_2pt_flux_interp_vol_ext[istate][Hadamard_columns_sparsity_ext[column_index]] 
                        += surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad][iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                }
            }
        }
    }//end of if split form or curvilinear split form



    // Evaluate reference numerical fluxes.
    
    std::array<std::vector<real>,nstate> conv_num_flux_dot_n;
    std::array<std::vector<real>,nstate> diss_auxi_num_flux_dot_n;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }

        std::array<real,nstate> entropy_var_face_int;
        std::array<real,nstate> entropy_var_face_ext;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_int;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_ext;
        std::array<real,nstate> soln_interp_to_face_int;
        std::array<real,nstate> soln_interp_to_face_ext;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state_int;
        std::array<dealii::Tensor<1,dim,real>,nstate> filtered_aux_soln_state_ext;
        std::array<real,nstate> filtered_soln_interp_to_face_int;
        std::array<real,nstate> filtered_soln_interp_to_face_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_interp_to_face_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_interp_to_face_int[istate] = legendre_soln_at_surf_q_int[istate][iquad];
            if(this->do_compute_filtered_solution) filtered_soln_interp_to_face_ext[istate] = legendre_soln_at_surf_q_ext[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad];
            entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q_int[istate][idim][iquad];
                aux_soln_state_ext[istate][idim] = aux_soln_at_surf_q_ext[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state_int[istate][idim] = legendre_aux_soln_at_surf_q_int[istate][idim][iquad];
                if(this->do_compute_filtered_solution) filtered_aux_soln_state_ext[istate][idim] = legendre_aux_soln_at_surf_q_ext[istate][idim][iquad];
            }
        }

        std::array<real,nstate> soln_state_int;
        soln_state_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
        std::array<real,nstate> soln_state_ext;
        soln_state_ext = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);


        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
                soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            }
        }

        // numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 
        // Note that the facet determinant of metric jacobian is the above norm multiplied by the determinant of the metric Jacobian evaluated on the facet.
        // Since the determinant of the metric Jacobian evaluated on the face cancels off, we can just scale the numerical flux by the norm.

        std::array<real,nstate> conv_num_flux_dot_n_at_q;
        std::array<real,nstate> diss_auxi_num_flux_dot_n_at_q;
        // Convective numerical flux. 
        conv_num_flux_dot_n_at_q = this->conv_num_flux_double->evaluate_flux(soln_state_int, soln_state_ext, unit_phys_normal_int);
        // dissipative numerical flux
        diss_auxi_num_flux_dot_n_at_q = this->diss_num_flux_double->evaluate_auxiliary_flux(
            current_cell_index, neighbor_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_interp_to_face_ext,
            aux_soln_state_int, aux_soln_state_ext,
            filtered_soln_interp_to_face_int, filtered_soln_interp_to_face_ext,
            filtered_aux_soln_state_int, filtered_aux_soln_state_ext,
            unit_phys_normal_int, penalty, false);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors of n_face_quad_pts.

            // allocate
            if(iquad == 0){
                conv_num_flux_dot_n[istate].resize(n_face_quad_pts);
                diss_auxi_num_flux_dot_n[istate].resize(n_face_quad_pts);
            }

            // write data
            conv_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
            diss_auxi_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate];
        }
    }

    // Compute RHS
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();
    for(int istate=0; istate<nstate; istate++){
        // interior RHS
        std::vector<real> rhs_int(n_shape_fns_int);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_int.inner_product_surface_1D(iface, 
                                                    surf_vol_ref_2pt_flux_interp_surf_int[istate], 
                                                    ones_surf, rhs_int, 
                                                    soln_basis_int.oneD_surf_operator, 
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol_int, 1.0);
            soln_basis_int.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_int[istate], 
                                            ones_vol, rhs_int, 
                                            soln_basis_int.oneD_vol_operator, 
                                            true, -1.0);
        }
        else 
        {
            soln_basis_int.inner_product_surface_1D(iface, 
                                                    conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                    surf_quad_weights, rhs_int, 
                                                    soln_basis_int.oneD_surf_operator, 
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, 1.0);
        }
        // dissipative flux
        soln_basis_int.inner_product_surface_1D(iface, 
                                                diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, 1.0);//adding=true, subtract the negative so add it
        // convective numerical flux
        soln_basis_int.inner_product_surface_1D(iface, conv_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        // dissipative numerical flux
        soln_basis_int.inner_product_surface_1D(iface, diss_auxi_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it


        for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
            local_rhs_int_cell(istate*n_shape_fns_int + ishape) += rhs_int[ishape];
        }

        // exterior RHS
        std::vector<real> rhs_ext(n_shape_fns_ext);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    surf_vol_ref_2pt_flux_interp_surf_ext[istate], 
                                                    ones_surf, rhs_ext, 
                                                    soln_basis_ext.oneD_surf_operator, 
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, -1.0);//the negative sign is bc the surface Hadamard function computes it on the otherside.
                                                    //to satisfy the unit test that checks consistency with Jesse Chan's formulation.
            std::vector<real> ones_vol(n_quad_pts_vol_ext, 1.0);
            soln_basis_ext.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_ext[istate], 
                                            ones_vol, rhs_ext, 
                                            soln_basis_ext.oneD_vol_operator, 
                                            true, -1.0);
        }
        else 
        {
            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                    surf_quad_weights, rhs_ext, 
                                                    soln_basis_ext.oneD_surf_operator, 
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, 1.0);//adding false
        }
        // dissipative flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true
        // convective numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, conv_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it
        // dissipative numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, diss_auxi_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it


        for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
            local_rhs_ext_cell(istate*n_shape_fns_ext + ishape) += rhs_ext[ishape];
        }
    }
}


/*******************************************************************
 *
 *
 *              PRIMARY EQUATIONS
 *
 *              NOTE: the implicit functions have not been modified.
 *              
 *              EVERYTHING BELOW Untouched/unverified/not used anymore
 *
 *******************************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index /*current_cell_index*/,
    const unsigned int ,//face_number,
    const unsigned int /*boundary_id*/,
    const dealii::FEFaceValuesBase<dim,dim> &/*fe_values_boundary*/,
    const real /*penalty*/,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim-1> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &/*soln_dof_indices*/,
    dealii::Vector<real> &/*local_rhs_int_cell*/,
    const bool /*compute_dRdW*/,
    const bool /*compute_dRdX*/,
    const bool /*compute_d2R*/)
{ 
    //Do nothing

//    (void) current_cell_index;
//    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
//    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
//    using ADArray = std::array<FadType,nstate>;
//    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;
// 
//    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
//    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;
// 
//    AssertDimension (n_dofs_cell, soln_dof_indices.size());
// 
//    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
//    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();
// 
//    std::vector<real> residual_derivatives(n_dofs_cell);
// 
//    std::vector<ADArray> soln_int(n_face_quad_pts);
//    std::vector<ADArray> soln_ext(n_face_quad_pts);
// 
//    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
//    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);
// 
//    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
//    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
//    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
//    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_int
//    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*
// 
//    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);
// 
//    // AD variable
//    std::vector< FadType > soln_coeff_int(n_dofs_cell);
//    const unsigned int n_total_indep = n_dofs_cell;
//    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
//        soln_coeff_int[idof] = this->solution(soln_dof_indices[idof]);
//        soln_coeff_int[idof].diff(idof, n_total_indep);
//    }
// 
//    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
//        for (int istate=0; istate<nstate; istate++) { 
//            // Interpolate solution to the face quadrature points
//            soln_int[iquad][istate]      = 0;
//            soln_grad_int[iquad][istate] = 0;
//        }
//    }
//    // Interpolate solution to face
//    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
//    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
// 
//        const dealii::Tensor<1,dim,FadType> normal_int = normals[iquad];
//        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;
// 
//        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
//            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
//            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
//            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
//        }
// 
//        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
//        dealii::Point<dim,FadType> ad_point;
//        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
//        this->pde_physics_fad->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
// 
//        //
//        // Evaluate physical convective flux, physical dissipative flux
//        // Following the the boundary treatment given by 
//        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
//        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
//        //      Details given on page 93
//        //conv_num_flux_dot_n[iquad] = this->conv_num_flux_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
// 
//        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
//        // Changing it back to the standdard F* = F*(Uin, Ubc)
//        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
//        // Losing 2p+1 OOA on functionals for all PDEs.
//        conv_num_flux_dot_n[iquad] = this->conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
// 
//        // Used for strong form
//        // Which physical convective flux to use?
//        conv_phys_flux[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
// 
//        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
//        diss_soln_num_flux[iquad] = this->diss_num_flux_fad->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
// 
//        ADArrayTensor1 diss_soln_jump_int;
//        ADArrayTensor1 diss_soln_jump_ext;
//        for (int s=0; s<nstate; s++) {
//            for (int d=0; d<dim; d++) {
//                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
//                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
//            }
//        }
//        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int, current_cell_index);
//        diss_flux_jump_ext[iquad] = this->pde_physics_fad->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext, neighbor_cell_index);
//
//        diss_auxi_num_flux_dot_n[iquad] = this->diss_num_flux_fad->evaluate_auxiliary_flux(
//            current_cell_index, neighbor_cell_index,
//            0.0, 0.0,
//            soln_int[iquad], soln_ext[iquad],
//            soln_grad_int[iquad], soln_grad_ext[iquad],
//            normal_int, penalty, true);
//    }
// 
//    // Boundary integral
//    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
// 
//        FadType rhs = 0.0;
// 
//        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;
// 
//        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
// 
//            // Convection
//            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
//            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
//            // Diffusive
//            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
//            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
//        }
//        // *******************
// 
//        local_rhs_int_cell(itest) += rhs.val();
// 
//        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
//            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
//                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
//                residual_derivatives[idof] = rhs.fastAccessDx(idof);
//            }
//            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
//        }
//    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index /*current_cell_index*/,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &/*cell_dofs_indices*/,
    dealii::Vector<real> &/*local_rhs_int_cell*/,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
    const bool /*compute_dRdW*/,
    const bool /*compute_dRdX*/,
    const bool /*compute_d2R*/)
{
    //Do nothing

//    (void) current_cell_index;
//    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
//    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
//    using ADArray = std::array<FadType,nstate>;
//    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;
//
//    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
//    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;
//
//    AssertDimension (n_dofs_cell, cell_dofs_indices.size());
//
//    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();
//
//    std::vector<real> residual_derivatives(n_dofs_cell);
//
//    std::vector< ADArray > soln_at_q(n_quad_pts);
//    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros
//
//    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
//    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
//    std::vector< ADArray > source_at_q(n_quad_pts);
//
//    // AD variable
//    std::vector< FadType > soln_coeff(n_dofs_cell);
//    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
//        soln_coeff[idof] = this->solution(cell_dofs_indices[idof]);
//        soln_coeff[idof].diff(idof, n_dofs_cell);
//    }
//    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
//        for (int istate=0; istate<nstate; istate++) { 
//            // Interpolate solution to the volume quadrature points
//            soln_at_q[iquad][istate]      = 0;
//            soln_grad_at_q[iquad][istate] = 0;
//        }
//    }
//    // Interpolate solution to face
//    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
//        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
//              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
//              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
//              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
//        }
//        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
//        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
//        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
//        // Evaluate physical convective flux and source term
//        conv_phys_flux_at_q[iquad] = this->pde_physics_double->convective_flux (soln_at_q[iquad]);
//        diss_phys_flux_at_q[iquad] = this->pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad], current_cell_index);
//        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
//            source_at_q[iquad] = this->pde_physics_double->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad], current_cell_index, this->current_time);
//        }
//    }
//
//
//    // Evaluate flux divergence by interpolating the flux
//    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
//    //const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();
//    std::vector<ADArray> flux_divergence(n_quad_pts);
//
//    std::array<std::array<std::vector<FadType>,nstate>,dim> f;
//    std::array<std::array<std::vector<FadType>,nstate>,dim> g;
//
//    for (int istate = 0; istate<nstate; ++istate) {
//        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
//            flux_divergence[iquad][istate] = 0.0;
//            for ( unsigned int flux_basis = 0; flux_basis < n_quad_pts; ++flux_basis ) {
//                flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad);
//            }
//
//        }
//    }
//
//    // Strong form
//    // The right-hand side sends all the term to the side of the source term
//    // Therefore, 
//    // \divergence ( Fconv + Fdiss ) = source 
//    // has the right-hand side
//    // rhs = - \divergence( Fconv + Fdiss ) + source 
//    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
//    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
//    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
//
//        FadType rhs = 0;
//
//
//        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;
//
//        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
//
//            // Convective
//            // Now minus such 2 integrations by parts
//            assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);
//
//            rhs = rhs - fe_values_vol.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW[iquad];
//
//            //// Diffusive
//            //// Note that for diffusion, the negative is defined in the physics
//            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
//            // Source
//
//            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
//                rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
//            }
//        }
//        //local_rhs_int_cell(itest) += rhs;
//
//    }
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator    /*cell*/,
    const dealii::types::global_dof_index                     /*current_cell_index*/,
    const dealii::types::global_dof_index                     /*neighbor_cell_index*/,
    const std::pair<unsigned int, int>                        /*face_subface_int*/,
    const std::pair<unsigned int, int>                        /*face_subface_ext*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_int*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_ext*/,
    const dealii::FEFaceValuesBase<dim,dim>                     &/*fe_values_int*/,
    const dealii::FEFaceValuesBase<dim,dim>     &/*fe_values_ext*/,
    const real /*penalty*/,
    const dealii::FESystem<dim,dim> &,//fe_int,
    const dealii::FESystem<dim,dim> &,//fe_ext,
    const dealii::Quadrature<dim-1> &,//face_quadrature_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &/*soln_dof_indices_int*/,
    const std::vector<dealii::types::global_dof_index> &/*soln_dof_indices_ext*/,
    dealii::Vector<real>          &/*local_rhs_int_cell*/,
    dealii::Vector<real>          &/*local_rhs_ext_cell*/,
    const bool /*compute_dRdW*/,
    const bool /*compute_dRdX*/,
    const bool /*compute_d2R*/)
{
    //Do nothing

//    (void) current_cell_index;
//    (void) neighbor_cell_index;
//    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
//    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
//    using ADArray = std::array<FadType,nstate>;
//    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;
//
//    // Use quadrature points of neighbor cell
//    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
//    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;
//
//    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
//    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;
//
//    AssertDimension (n_dofs_int, soln_dof_indices_int.size());
//    AssertDimension (n_dofs_ext, soln_dof_indices_ext.size());
//
//    // Jacobian and normal should always be consistent between two elements
//    // even for non-conforming meshes?
//    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
//    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();
//
//    // AD variable
//    std::vector<FadType> soln_coeff_int_ad(n_dofs_int);
//    std::vector<FadType> soln_coeff_ext_ad(n_dofs_ext);
//
//
//    // Jacobian blocks
//    std::vector<real> dR1_dW1(n_dofs_int);
//    std::vector<real> dR1_dW2(n_dofs_ext);
//    std::vector<real> dR2_dW1(n_dofs_int);
//    std::vector<real> dR2_dW2(n_dofs_ext);
//
//    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
//    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
//    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);
//
//    // Interpolate solution to the face quadrature points
//    std::vector< ADArray > soln_int(n_face_quad_pts);
//    std::vector< ADArray > soln_ext(n_face_quad_pts);
//
//    std::vector< ADArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
//    std::vector< ADArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros
//
//    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
//    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*
//
//    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
//    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
//    // AD variable
//    const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
//    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
//        soln_coeff_int_ad[idof] = this->solution(soln_dof_indices_int[idof]);
//        soln_coeff_int_ad[idof].diff(idof, n_total_indep);
//    }
//    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
//        soln_coeff_ext_ad[idof] = this->solution(soln_dof_indices_ext[idof]);
//        soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);
//    }
//    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
//        for (int istate=0; istate<nstate; istate++) { 
//            soln_int[iquad][istate]      = 0;
//            soln_grad_int[iquad][istate] = 0;
//            soln_ext[iquad][istate]      = 0;
//            soln_grad_ext[iquad][istate] = 0;
//        }
//    }
//    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
//
//        const dealii::Tensor<1,dim,FadType> normal_int = normals_int[iquad];
//        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;
//
//        // Interpolate solution to face
//        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
//            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
//            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
//            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
//        }
//        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
//            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
//            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
//            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
//        }
//        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
//        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
//        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
//        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
//        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
//        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;
//
//        // Evaluate physical convective flux, physical dissipative flux, and source term
//        conv_num_flux_dot_n[iquad] = this->conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
//
//        conv_phys_flux_int[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
//        conv_phys_flux_ext[iquad] = this->pde_physics_fad->convective_flux (soln_ext[iquad]);
//
//        diss_soln_num_flux[iquad] = this->diss_num_flux_fad->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);
//
//        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
//        for (int s=0; s<nstate; s++) {
//            for (int d=0; d<dim; d++) {
//                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
//                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
//            }
//        }
//        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int, current_cell_index);
//        diss_flux_jump_ext[iquad] = this->pde_physics_fad->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext, neighbor_cell_index);
//
//        diss_auxi_num_flux_dot_n[iquad] = this->diss_num_flux_fad->evaluate_auxiliary_flux(
//            current_cell_index, neighbor_cell_index,
//            0.0, 0.0,
//            soln_int[iquad], soln_ext[iquad],
//            soln_grad_int[iquad], soln_grad_ext[iquad],
//            normal_int, penalty);
//    }
//
//    // From test functions associated with interior cell point of view
//    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
//        FadType rhs = 0.0;
//        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;
//
//        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
//            // Convection
//            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux_int[iquad][istate]*normals_int[iquad];
//            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad];
//            // Diffusive
//            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
//            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
//        }
//
//        local_rhs_int_cell(itest_int) += rhs.val();
//        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
//            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
//                dR1_dW1[idof] = rhs.fastAccessDx(idof);
//            }
//            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
//                dR1_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
//            }
//            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, dR1_dW1);
//            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, dR1_dW2);
//        }
//    }
//
//    // From test functions associated with neighbour cell point of view
//    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
//        FadType rhs = 0.0;
//        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;
//
//        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
//            // Convection
//            const FadType flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_phys_flux_ext[iquad][istate]*(-normals_int[iquad]);
//            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad];
//            // Diffusive
//            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
//            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
//        }
//
//        local_rhs_ext_cell(itest_ext) += rhs.val();
//        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
//            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
//                dR2_dW1[idof] = rhs.fastAccessDx(idof);
//            }
//            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
//                dR2_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
//            }
//            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, dR2_dW1);
//            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, dR2_dW2);
//        }
//    }
}

/*******************************************************
 *
 *                     EXPLICIT
 *
 *******************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index /*current_cell_index*/,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const std::vector<dealii::types::global_dof_index> &/*cell_dofs_indices*/,
    const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices*/,
    const unsigned int /*poly_degree*/,
    const unsigned int /*grid_degree*/,
    dealii::Vector<real> &/*local_rhs_int_cell*/,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    //do nothing
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::allocate_dual_vector()
{
    //Do nothing.
}

// using default MeshType = Triangulation
// 1D: dealii::Triangulation<dim>;
// Otherwise: dealii::parallel::distributed::Triangulation<dim>;
template class DGStrong <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::Triangulation<PHILIP_DIM>>;

template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
