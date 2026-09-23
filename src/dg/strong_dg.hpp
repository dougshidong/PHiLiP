#ifndef __STRONG_DISCONTINUOUSGALERKIN_H__
#define __STRONG_DISCONTINUOUSGALERKIN_H__

#include "dg_base_state.hpp"

namespace PHiLiP {

/// DGStrong class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nspecies, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nspecies, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGStrong: public DGBaseState<dim, nspecies, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGBaseState<dim,nspecies,nstate,real,MeshType>::Triangulation;

public:
    /// Constructor
    DGStrong(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    const bool do_compute_filtered_solution; ///< Flag to compute the filtered solution
    const bool apply_modal_high_pass_filter_on_filtered_solution; ///< Flag to apply modal high pass filter on the filtered solution
    const unsigned int poly_degree_max_large_scales; ///< For filtered solution; lower bound of high pass filter
    const bool using_wall_model; ///< Flag for using wall model
    const bool wall_model_input_from_second_element; /// Flag for using the second element as the wall model input
    const bool uncollocated_flux_nodes; /// Flag for using GL flux nodes

    /// Assembles the auxiliary equations' residuals and solves for the auxiliary variables.
    /** For information regarding auxiliary vs. primary quations, see 
     *  Quaegebeur, Nadarajah, Navah and Zwanenburg 2019: Stability of Energy Stable Flux 
     *                Reconstruction for the Diffusion Problem Using Compact Numerical Fluxes
     */
    void assemble_auxiliary_residual (const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Allocate the dual vector for optimization.
    void allocate_dual_vector (const bool compute_d2R);

private:
    /// Assembles the auxiliary equations' cell residuals.
    template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
    void assemble_cell_auxiliary_residual (
        const DoFCellAccessorType1 &current_cell,
        const DoFCellAccessorType2 &current_metric_cell,
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rhs);

protected:
    /// Builds the necessary operators and assembles volume residual for either primary or auxiliary.
    template <typename adtype>
    void assemble_volume_term_and_build_operators_ad_templated(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<adtype>                              &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<adtype>>        &aux_soln_coeffs,
        const std::vector<adtype>                              &metric_coeffs,
        const std::vector<real>                                &local_dual,
        const std::vector<dealii::types::global_dof_index>     &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dofs_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>    &physics,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>           &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<adtype>,dim>                    &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume*/,
        dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume_lagrange*/,
        const dealii::FESystem<dim,dim>                        &/*fe_soln*/,
        std::vector<adtype>                                    &rhs, 
        dealii::Tensor<1,dim,std::vector<adtype>>              &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        adtype                                                 &dual_dot_residual);
    
    /// Calls the function to assemble volume residual. For double type.
    void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<double>                              &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<double>>        &aux_soln_coeffs,
        const std::vector<double>                              &metric_coeffs,
        const std::vector<real>                                &local_dual,
        const std::vector<dealii::types::global_dof_index>     &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dofs_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<double,dim,2*dim>           &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<double>,dim>                    &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                        &fe_soln,
        std::vector<double>                                    &rhs, 
        dealii::Tensor<1,dim,std::vector<double>>              &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        double                                                 &dual_dot_residual)  override
    {
        assemble_volume_term_and_build_operators_ad_templated<double>(
                cell,
                current_cell_index,
                soln_coeffs,
                aux_soln_coeffs,
                metric_coeffs,
                local_dual,
                soln_dofs_indices,
                metric_dofs_indices,
                poly_degree,
                grid_degree,
                *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_double),
                soln_basis,
                flux_basis,
                flux_basis_stiffness,
                soln_basis_projection_oper_int,
                soln_basis_projection_oper_ext,
                metric_oper,
                mapping_basis,
                mapping_support_points,
                fe_values_collection_volume,
                fe_values_collection_volume_lagrange,
                fe_soln,
                rhs,
                local_auxiliary_RHS,
                compute_auxiliary_right_hand_side,
                dual_dot_residual);
    }
    
    /// Calls the function to assemble volume residual. For codi_JacobianComputationType.
    void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator                cell,
        const dealii::types::global_dof_index                                 current_cell_index,
        const std::vector<codi_JacobianComputationType>                       &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>> &aux_soln_coeffs,
        const std::vector<codi_JacobianComputationType>                       &metric_coeffs,
        const std::vector<real>                                               &local_dual,
        const std::vector<dealii::types::global_dof_index>                    &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>                    &metric_dofs_indices,
        const unsigned int                                                    poly_degree,
        const unsigned int                                                    grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                                  &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                                  &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>                            &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                          &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                          &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim>    &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                          &mapping_basis,
        std::array<std::vector<codi_JacobianComputationType>,dim>             &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                                         &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                                         &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                                       &fe_soln,
        std::vector<codi_JacobianComputationType>                             &rhs, 
        dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>       &local_auxiliary_RHS,
        const bool                                                            compute_auxiliary_right_hand_side,
        codi_JacobianComputationType                                          &dual_dot_residual)  override
    {
        assemble_volume_term_and_build_operators_ad_templated<codi_JacobianComputationType>(
                cell,
                current_cell_index,
                soln_coeffs,
                aux_soln_coeffs,
                metric_coeffs,
                local_dual,
                soln_dofs_indices,
                metric_dofs_indices,
                poly_degree,
                grid_degree,
                *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad),
                soln_basis,
                flux_basis,
                flux_basis_stiffness,
                soln_basis_projection_oper_int,
                soln_basis_projection_oper_ext,
                metric_oper,
                mapping_basis,
                mapping_support_points,
                fe_values_collection_volume,
                fe_values_collection_volume_lagrange,
                fe_soln,
                rhs,
                local_auxiliary_RHS,
                compute_auxiliary_right_hand_side,
                dual_dot_residual);
    }
    
    /// Calls the function to assemble volume residual. For codi_HessianComputationType.
    void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator               cell,
        const dealii::types::global_dof_index                                current_cell_index,
        const std::vector<codi_HessianComputationType>                       &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>> &aux_soln_coeffs,
        const std::vector<codi_HessianComputationType>                       &metric_coeffs,
        const std::vector<real>                                              &local_dual,
        const std::vector<dealii::types::global_dof_index>                   &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>                   &metric_dofs_indices,
        const unsigned int                                                   poly_degree,
        const unsigned int                                                   grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                                 &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                                 &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>                           &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                         &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                         &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim>    &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                         &mapping_basis,
        std::array<std::vector<codi_HessianComputationType>,dim>             &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                                        &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                                        &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                                      &fe_soln,
        std::vector<codi_HessianComputationType>                             &rhs, 
        dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>       &local_auxiliary_RHS,
        const bool                                                           compute_auxiliary_right_hand_side,
        codi_HessianComputationType                                          &dual_dot_residual)  override
    {
        assemble_volume_term_and_build_operators_ad_templated<codi_HessianComputationType>(
                cell,
                current_cell_index,
                soln_coeffs,
                aux_soln_coeffs,
                metric_coeffs,
                local_dual,
                soln_dofs_indices,
                metric_dofs_indices,
                poly_degree,
                grid_degree,
                *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad_fad),
                soln_basis,
                flux_basis,
                flux_basis_stiffness,
                soln_basis_projection_oper_int,
                soln_basis_projection_oper_ext,
                metric_oper,
                mapping_basis,
                mapping_support_points,
                fe_values_collection_volume,
                fe_values_collection_volume_lagrange,
                fe_soln,
                rhs,
                local_auxiliary_RHS,
                compute_auxiliary_right_hand_side,
                dual_dot_residual);
    }

    /// Builds the necessary operators and assembles boundary residual for either primary or auxiliary.
    template <typename adtype>
    void assemble_boundary_term_and_build_operators_ad_templated(
        typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
        const dealii::types::global_dof_index                              current_cell_index,
        const std::vector<adtype>                                          &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &aux_soln_coeffs,
        const std::vector<adtype>                                          &metric_coeffs,
        const std::vector<real>                                            &local_dual,
        const unsigned int                                                 face_number,
        const unsigned int                                                 boundary_id,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>                          &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nspecies, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        const unsigned int                                                 poly_degree,
        const unsigned int                                                 grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
        std::array<std::vector<adtype>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_int*/,
        const dealii::FESystem<dim,dim>                                    &/*fe_soln*/,
        const real                                                         penalty,
        std::vector<adtype>                                                &rhs,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &local_auxiliary_RHS,
        const bool                                                         compute_auxiliary_right_hand_side,
        adtype                                                             &dual_dot_residual);
    
    /// Calls the function to assemble boundary residual. For double type.
    void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
        const dealii::types::global_dof_index                              current_cell_index,
        const std::vector<double>                                          &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<double>>                    &aux_soln_coeffs,
        const std::vector<double>                                          &metric_coeffs,
        const std::vector<real>                                            &local_dual,
        const unsigned int                                                 face_number,
        const unsigned int                                                 boundary_id,
        const unsigned int                                                 poly_degree,
        const unsigned int                                                 grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
        OPERATOR::metric_operators<double,dim,2*dim>                       &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
        std::array<std::vector<double>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                                    &fe_soln,
        const real                                                         penalty,
        std::vector<double>                                                &rhs,
        dealii::Tensor<1,dim,std::vector<double>>                          &local_auxiliary_RHS,
        const bool                                                         compute_auxiliary_right_hand_side,
        double                                                             &dual_dot_residual) override
    {
        assemble_boundary_term_and_build_operators_ad_templated<double>(
            cell, 
            current_cell_index,
            soln_coeffs,
            aux_soln_coeffs,
            metric_coeffs,
            local_dual,
            face_number,
            boundary_id,
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_double),
            poly_degree,
            grid_degree,
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            mapping_basis,
            mapping_support_points,
            fe_values_collection_face_int,
            fe_soln,
            penalty,
            rhs,
            local_auxiliary_RHS,
            compute_auxiliary_right_hand_side,
            dual_dot_residual);
    }
    
    /// Calls the function to assemble boundary residual. For codi_JacobianComputationType.
    void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator                                   cell,
        const dealii::types::global_dof_index                                                    current_cell_index,
        const std::vector<codi_JacobianComputationType>                                          &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                    &aux_soln_coeffs,
        const std::vector<codi_JacobianComputationType>                                          &metric_coeffs,
        const std::vector<real>                                                                  &local_dual,
        const unsigned int                                                                       face_number,
        const unsigned int                                                                       boundary_id,
        const unsigned int                                                                       poly_degree,
        const unsigned int                                                                       grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                                                     &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                                                     &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>                                             &soln_basis_projection_oper_int,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim>                       &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                                             &mapping_basis,
        std::array<std::vector<codi_JacobianComputationType>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                                        &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                                                          &fe_soln,
        const real                                                                               penalty,
        std::vector<codi_JacobianComputationType>                                                &rhs,
        dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                          &local_auxiliary_RHS,
        const bool                                                                               compute_auxiliary_right_hand_side,
        codi_JacobianComputationType                                                             &dual_dot_residual) override
    {
        assemble_boundary_term_and_build_operators_ad_templated<codi_JacobianComputationType>(
            cell, 
            current_cell_index,
            soln_coeffs,
            aux_soln_coeffs,
            metric_coeffs,
            local_dual,
            face_number,
            boundary_id,
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_rad),
            poly_degree,
            grid_degree,
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            mapping_basis,
            mapping_support_points,
            fe_values_collection_face_int,
            fe_soln,
            penalty,
            rhs,
            local_auxiliary_RHS,
            compute_auxiliary_right_hand_side,
            dual_dot_residual);
    }
    
    /// Calls the function to assemble boundary residual. For codi_HessianComputationType.
    void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator                                  cell,
        const dealii::types::global_dof_index                                                   current_cell_index,
        const std::vector<codi_HessianComputationType>                                          &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                    &aux_soln_coeffs,
        const std::vector<codi_HessianComputationType>                                          &metric_coeffs,
        const std::vector<real>                                                                 &local_dual,
        const unsigned int                                                                      face_number,
        const unsigned int                                                                      boundary_id,
        const unsigned int                                                                      poly_degree,
        const unsigned int                                                                      grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                                                    &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                                                    &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>                                            &soln_basis_projection_oper_int,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim>                       &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                                            &mapping_basis,
        std::array<std::vector<codi_HessianComputationType>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                                       &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                                                         &fe_soln,
        const real                                                                              penalty,
        std::vector<codi_HessianComputationType>                                                &rhs,
        dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                          &local_auxiliary_RHS,
        const bool                                                                              compute_auxiliary_right_hand_side,
        codi_HessianComputationType                                                             &dual_dot_residual) override
    {
        assemble_boundary_term_and_build_operators_ad_templated<codi_HessianComputationType>(
            cell, 
            current_cell_index,
            soln_coeffs,
            aux_soln_coeffs,
            metric_coeffs,
            local_dual,
            face_number,
            boundary_id,
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_rad_fad),
            poly_degree,
            grid_degree,
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            mapping_basis,
            mapping_support_points,
            fe_values_collection_face_int,
            fe_soln,
            penalty,
            rhs,
            local_auxiliary_RHS,
            compute_auxiliary_right_hand_side,
            dual_dot_residual);
    }
    
    /// Calls the function to assemble face residual.
    template <typename adtype>
    void assemble_face_term_and_build_operators_ad_templated(
        typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator             neighbor_cell,
        const dealii::types::global_dof_index                              current_cell_index,
        const dealii::types::global_dof_index                              neighbor_cell_index,
        const unsigned int                                                 iface,
        const unsigned int                                                 neighbor_iface,
        const std::vector<adtype>                                          &soln_coeffs_int,
        const std::vector<adtype>                                          &soln_coeffs_ext,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &aux_soln_coeffs_int,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &aux_soln_coeffs_ext,
        const std::vector<adtype>                                          &metric_coeff_int,
        const std::vector<adtype>                                          &metric_coeff_ext,
        const std::vector< double >                                        &dual_int,
        const std::vector< double >                                        &dual_ext,
        const unsigned int                                                 poly_degree_int,
        const unsigned int                                                 poly_degree_ext,
        const unsigned int                                                 grid_degree_int,
        const unsigned int                                                 grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>                         &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
        std::array<std::vector<adtype>,dim>                                &mapping_support_points,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>                          &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nspecies, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_int*/,
        dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_ext*/,
        dealii::hp::FESubfaceValues<dim,dim>                               &/*fe_values_collection_subface*/,
        const dealii::FESystem<dim,dim>                                    &/*fe_int*/,
        const dealii::FESystem<dim,dim>                                    &/*fe_ext*/,
        const real                                                         penalty,
        std::vector<adtype>                                                &rhs_int,
        std::vector<adtype>                                                &rhs_ext,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &aux_rhs_int,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &aux_rhs_ext,
        const bool                                                         compute_auxiliary_right_hand_side,
        adtype                                                             &dual_dot_residual,
        const bool                                                         /*is_a_subface*/,
        const unsigned int                                                 /*neighbor_i_subface*/);
    
    /// Calls the function to assemble face residual. For double type.
    void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator             neighbor_cell,
        const dealii::types::global_dof_index                              current_cell_index,
        const dealii::types::global_dof_index                              neighbor_cell_index,
        const unsigned int                                                 iface,
        const unsigned int                                                 neighbor_iface,
        const std::vector<double>                                          &soln_coeff_int,
        const std::vector<double>                                          &soln_coeff_ext,
        const dealii::Tensor<1,dim,std::vector<double>>                    &aux_soln_coeff_int,
        const dealii::Tensor<1,dim,std::vector<double>>                    &aux_soln_coeff_ext,
        const std::vector<double>                                          &metric_coeff_int,
        const std::vector<double>                                          &metric_coeff_ext,
        const std::vector< double >                                        &dual_int,
        const std::vector< double >                                        &dual_ext,
        const unsigned int                                                 poly_degree_int,
        const unsigned int                                                 poly_degree_ext,
        const unsigned int                                                 grid_degree_int,
        const unsigned int                                                 grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>                         &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<double,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<double,dim,2*dim>                       &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
        std::array<std::vector<double>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                               &fe_values_collection_subface,
        const dealii::FESystem<dim,dim>                                    &fe_int,
        const dealii::FESystem<dim,dim>                                    &fe_ext,
        const real                                                         penalty,
        std::vector<double>                                                &rhs_int,
        std::vector<double>                                                &rhs_ext,
        dealii::Tensor<1,dim,std::vector<double>>                          &aux_rhs_int,
        dealii::Tensor<1,dim,std::vector<double>>                          &aux_rhs_ext,
        const bool                                                         compute_auxiliary_right_hand_side,
        double                                                             &dual_dot_residual,
        const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/,
        const bool                                                         is_a_subface,
        const unsigned int                                                 neighbor_i_subface) override
    {
        assemble_face_term_and_build_operators_ad_templated<double>(
            cell,
            neighbor_cell,
            current_cell_index,
            neighbor_cell_index,
            iface,
            neighbor_iface,
            soln_coeff_int,
            soln_coeff_ext,
            aux_soln_coeff_int,
            aux_soln_coeff_ext,
            metric_coeff_int,
            metric_coeff_ext,
            dual_int,
            dual_ext,
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
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_double),
            fe_values_collection_face_int,
            fe_values_collection_face_ext,
            fe_values_collection_subface,
            fe_int,
            fe_ext,
            penalty,
            rhs_int,
            rhs_ext,
            aux_rhs_int,
            aux_rhs_ext,
            compute_auxiliary_right_hand_side,
            dual_dot_residual,
            is_a_subface,
            neighbor_i_subface);
    }
 
    /// Calls the function to assemble face residual. For codi_JacobianComputationType.
    void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator                                   cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator                                   neighbor_cell,
        const dealii::types::global_dof_index                                                    current_cell_index,
        const dealii::types::global_dof_index                                                    neighbor_cell_index,
        const unsigned int                                                                       iface,
        const unsigned int                                                                       neighbor_iface,
        const std::vector<codi_JacobianComputationType>                                          &soln_coeff_int,
        const std::vector<codi_JacobianComputationType>                                          &soln_coeff_ext,
        const dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                    &aux_soln_coeff_int,
        const dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                    &aux_soln_coeff_ext,
        const std::vector<codi_JacobianComputationType>                                          &metric_coeff_int,
        const std::vector<codi_JacobianComputationType>                                          &metric_coeff_ext,
        const std::vector< double >                                                              &dual_int,
        const std::vector< double >                                                              &dual_ext,
        const unsigned int                                                                       poly_degree_int,
        const unsigned int                                                                       poly_degree_ext,
        const unsigned int                                                                       grid_degree_int,
        const unsigned int                                                                       grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>                                                     &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                                                     &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                                                     &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                                                     &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>                                               &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                                             &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                                             &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim>                       &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>                                             &mapping_basis,
        std::array<std::vector<codi_JacobianComputationType>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                                        &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                                        &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                                                     &fe_values_collection_subface,
        const dealii::FESystem<dim,dim>                                                          &fe_int,
        const dealii::FESystem<dim,dim>                                                          &fe_ext,
        const real                                                                               penalty,
        std::vector<codi_JacobianComputationType>                                                &rhs_int,
        std::vector<codi_JacobianComputationType>                                                &rhs_ext,
        dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                          &aux_rhs_int,
        dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>>                          &aux_rhs_ext,
        const bool                                                                               compute_auxiliary_right_hand_side,
        codi_JacobianComputationType                                                             &dual_dot_residual,
        const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/,
        const bool                                                                               is_a_subface,
        const unsigned int                                                                       neighbor_i_subface) override
    {
        assemble_face_term_and_build_operators_ad_templated<codi_JacobianComputationType>(
            cell,
            neighbor_cell,
            current_cell_index,
            neighbor_cell_index,
            iface,
            neighbor_iface,
            soln_coeff_int,
            soln_coeff_ext,
            aux_soln_coeff_int,
            aux_soln_coeff_ext,
            metric_coeff_int,
            metric_coeff_ext,
            dual_int,
            dual_ext,
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
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_rad),
            fe_values_collection_face_int,
            fe_values_collection_face_ext,
            fe_values_collection_subface,
            fe_int,
            fe_ext,
            penalty,
            rhs_int,
            rhs_ext,
            aux_rhs_int,
            aux_rhs_ext,
            compute_auxiliary_right_hand_side,
            dual_dot_residual,
            is_a_subface,
            neighbor_i_subface);
    }

    /// Calls the function to assemble face residual. For codi_HessianComputationType.
    void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator                                  cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator                                  neighbor_cell,
        const dealii::types::global_dof_index                                                   current_cell_index,
        const dealii::types::global_dof_index                                                   neighbor_cell_index,
        const unsigned int                                                                      iface,
        const unsigned int                                                                      neighbor_iface,
        const std::vector<codi_HessianComputationType>                                          &soln_coeff_int,
        const std::vector<codi_HessianComputationType>                                          &soln_coeff_ext,
        const dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                    &aux_soln_coeff_int,
        const dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                    &aux_soln_coeff_ext,
        const std::vector<codi_HessianComputationType>                                          &metric_coeff_int,
        const std::vector<codi_HessianComputationType>                                          &metric_coeff_ext,
        const std::vector< double >                                                             &dual_int,
        const std::vector< double >                                                             &dual_ext,
        const unsigned int                                                                      poly_degree_int,
        const unsigned int                                                                      poly_degree_ext,
        const unsigned int                                                                      grid_degree_int,
        const unsigned int                                                                      grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>                                                    &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                                                    &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                                                    &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                                                    &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>                                              &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>                                            &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                                            &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim>                       &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>                                            &mapping_basis,
        std::array<std::vector<codi_HessianComputationType>,dim>                                &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                                                       &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                                       &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                                                    &fe_values_collection_subface,
        const dealii::FESystem<dim,dim>                                                         &fe_int,
        const dealii::FESystem<dim,dim>                                                         &fe_ext,
        const real                                                                              penalty,
        std::vector<codi_HessianComputationType>                                                &rhs_int,
        std::vector<codi_HessianComputationType>                                                &rhs_ext,
        dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                          &aux_rhs_int,
        dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>>                          &aux_rhs_ext,
        const bool                                                                              compute_auxiliary_right_hand_side,
        codi_HessianComputationType                                                             &dual_dot_residual,
        const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/,
        const bool                                                                              is_a_subface,
        const unsigned int                                                                      neighbor_i_subface) override
    {
        assemble_face_term_and_build_operators_ad_templated<codi_HessianComputationType>(
            cell,
            neighbor_cell,
            current_cell_index,
            neighbor_cell_index,
            iface,
            neighbor_iface,
            soln_coeff_int,
            soln_coeff_ext,
            aux_soln_coeff_int,
            aux_soln_coeff_ext,
            metric_coeff_int,
            metric_coeff_ext,
            dual_int,
            dual_ext,
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
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nspecies,nstate,real,MeshType>::diss_num_flux_rad_fad),
            fe_values_collection_face_int,
            fe_values_collection_face_ext,
            fe_values_collection_subface,
            fe_int,
            fe_ext,
            penalty,
            rhs_int,
            rhs_ext,
            aux_rhs_int,
            aux_rhs_ext,
            compute_auxiliary_right_hand_side,
            dual_dot_residual,
            is_a_subface,
            neighbor_i_subface);
    }

public:
    ///Evaluate the volume RHS for the auxiliary equation.
    /** \f[
    * \int_{\mathbf{\Omega}_r} \chi_i(\mathbf{\xi}^r) \left( \nabla^r(u) \right)\mathbf{C}_m(\mathbf{\xi}^r) d\mathbf{\Omega}_r,\:\forall i=1,\dots,N_p.
    * \f]
    */
    template <typename adtype>
    void assemble_volume_term_auxiliary_equation(
        const std::array<std::vector<adtype>,nstate>  &soln_coeff,
        const unsigned int                            poly_degree,
        OPERATOR::basis_functions<dim,2*dim>          &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>          &flux_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>  &metric_oper,
        dealii::Tensor<1,dim,std::vector<adtype>>     &local_auxiliary_RHS);

protected:
    /// Evaluate the boundary RHS for the auxiliary equation.
    template <typename adtype>
    void assemble_boundary_term_auxiliary_equation(
        const unsigned int                                 iface,
        const dealii::types::global_dof_index              current_cell_index,
        std::vector<bool>                                  face_orientation,
        const std::array<std::vector<adtype>,nstate>       &soln_coeff,
        const unsigned int                                 poly_degree,
        const unsigned int                                 boundary_id,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>       &metric_oper,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>                    &pde_physics,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS);

public:
    /// Evaluate the facet RHS for the auxiliary equation.
    /** \f[
    * \int_{\mathbf{\Gamma}_r} \chi_i \left( \hat{\mathbf{n}}^r\mathbf{C}_m(\mathbf{\xi}^r)^T\right) \cdot \left[ u^*  
    * - u\right]d\mathbf{\Gamma}_r,\:\forall i=1,\dots,N_p.
    * \f]
    */
    template <typename adtype>
    void assemble_face_term_auxiliary_equation(
        const unsigned int                                 iface, 
        const unsigned int                                 neighbor_iface,
        const dealii::types::global_dof_index              current_cell_index,
        const dealii::types::global_dof_index              neighbor_cell_index,
        std::vector<bool>                                  face_orientation_int,
        std::vector<bool>                                  face_orientation_ext,
        const std::array<std::vector<adtype>,nstate>       &soln_coeff_int,
        const std::array<std::vector<adtype>,nstate>       &soln_coeff_ext,
        const unsigned int                                 poly_degree_int, 
        const unsigned int                                 poly_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>       &metric_oper_int,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>                    &pde_physics,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS_int,
        dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS_ext);

protected:
    ///Evaluate volume RHS for the viscous term.
    template <typename adtype>
    void assemble_volume_term_viscous_primal(
        const dealii::types::global_dof_index                              current_cell_index,
        const std::array<std::vector<adtype>,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_q,
        const std::array<std::vector<adtype>,nstate> &legendre_soln_at_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_q,
        const unsigned int                            poly_degree,
        OPERATOR::basis_functions<dim,2*dim>          &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>          &flux_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>  &metric_oper,
        const std::array<std::vector<adtype>,nstate>  &entropy_var_coeff,
        const std::array<std::vector<adtype>,nstate>  &entropy_var_at_q,
        const unsigned int  n_quad_pts,
        const unsigned int  n_dofs_cell,
        const unsigned int  n_shape_fns,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype> &pde_physics,
        std::vector<adtype>     &vol_term_viscous) const;
    
    ///Evaluate the boundary RHS for viscous term.
    template <typename adtype>
    void assemble_boundary_term_viscous_primal(
        const dealii::types::global_dof_index                              current_cell_index,
        std::vector<bool>                                                  face_orientation,
        const unsigned int                                                 iface,
        const unsigned int                                                 boundary_id,
        const unsigned int                                                 poly_degree,
        const real                                                         penalty,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_quads,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_quads,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_vol_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_vol_q,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_surf_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_surf_q,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_opposite_surf_q,
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_vol_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_vol_q,
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_surf_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_surf_q,
        const unsigned int                                                 n_vol_quad_pts,  
        const unsigned int                                                 n_face_quad_pts,  
        const unsigned int                                                 n_dofs_cell, 
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normals,
        const std::vector<adtype>                                          &JxW_face,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>          &pde_physics,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        std::vector<adtype>                                                &boundary_term_viscous) const;
    
    ///Evaluate the face RHS for viscous term.
    template <typename adtype>
    void assemble_face_term_viscous_primal(
        const dealii::types::global_dof_index                              current_cell_index,
        const dealii::types::global_dof_index                              neighbor_cell_index,
        std::vector<bool>                                                  face_orientation_int,
        std::vector<bool>                                                  face_orientation_ext,
        const unsigned int                                                 iface_int,
        const unsigned int                                                 iface_ext,
        const real                                                         penalty,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_ext,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_ext,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_ext,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_vol_q_int,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_vol_q_ext,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_vol_q_int,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_vol_q_ext,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_surf_q_int,
        const std::array<std::vector<adtype>,nstate>                       &soln_at_surf_q_ext,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_surf_q_int,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_surf_q_ext,
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_vol_q_int,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_vol_q_int,
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_vol_q_ext,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_vol_q_ext, 
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_surf_q_int,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_surf_q_int,
        const std::array<std::vector<adtype>,nstate>                       &legendre_soln_at_surf_q_ext,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_surf_q_ext,
        const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal_int,
        const std::vector<adtype>                                          &JxW_face,
        const unsigned int                                                  poly_degree_int,
        const unsigned int                                                  poly_degree_ext,
        const unsigned int                                                  n_quad_pts_vol_int,
        const unsigned int                                                  n_quad_pts_vol_ext,
        const unsigned int                                                  n_dofs_cell_int,
        const unsigned int                                                  n_dofs_cell_ext,
        const unsigned int                                                  n_face_quad_pts,
        const unsigned int                                                  n_shape_fns_int,
        const unsigned int                                                  n_shape_fns_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>          &pde_physics,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        std::vector<adtype>                                                &viscous_face_term_int,
        std::vector<adtype>                                                &viscous_face_term_ext) const;
    
    /// Compute filtered solution in the volume
    template <typename adtype>
    void compute_filtered_solution_volume(
        std::array<std::vector<adtype>,nstate> &legendre_soln_at_q,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_q,
        const unsigned int n_quad_pts,
        const unsigned int n_shape_fns,
        const unsigned int poly_degree,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype> &pde_physics,
        const std::array<std::vector<adtype>,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_q);
    
    /// Compute filtered solution at the face and volume
    template <typename adtype>
    void compute_filtered_solution_volume_and_face(
        std::vector<bool>                                                  face_orientation,
        const unsigned int                                                 iface,
        std::array<std::vector<adtype>,nstate> &legendre_soln_at_vol_q,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_vol_q,
        std::array<std::vector<adtype>,nstate> &legendre_soln_at_surf_q,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &legendre_aux_soln_at_surf_q,
        const std::array<std::vector<adtype>,nstate> &soln_at_vol_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_vol_q,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>                &pde_physics,
        const std::array<std::vector<adtype>,nstate> &soln_at_surf_q,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_at_surf_q,
        const unsigned int n_quad_pts_vol,
        const unsigned int n_face_quad_pts,
        const unsigned int n_shape_fns,
        const unsigned int poly_degree);

protected:
    /// Strong form primary equation's volume right-hand-side.
    /**  We refer to Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.
    * Conservative form Eq. (17): <br>
    *\f[ 
    * \int_{\mathbf{\Omega}_r} \chi_i (\mathbf{\xi}^r) \left(\nabla^r \phi(\mathbf{\xi}^r) \cdot \hat{\mathbf{f}}^r \right) d\mathbf{\Omega}_r
    * ,\:\forall i=1,\dots,N_p,
    * \f] 
    * where \f$ \chi \f$ is the basis function, \f$ \phi \f$ is the flux basis (basis collocated on the volume cubature nodes,
    * and \f$\hat{\mathbf{f}}^r = \mathbf{\Pi}\left(\mathbf{f}_m\mathbf{C}_m \right) \f$ is the projection of the reference flux.
    * <br> Entropy stable two-point flux form (extension of Eq. (22))<br>
    * \f[ 
    * \mathbf{\chi}(\mathbf{\xi}_v^r)^T \mathbf{W} 2 \left[ \nabla^r\mathbf{\phi}(\mathbf{\xi}_v^r) \circ
    * \mathbf{F}^r\right] \mathbf{1}^T d\mathbf{\Omega}_r
    * ,\:\forall i=1,\dots,N_p,
    * \f]
    * where \f$ (\mathbf{F})_{ij} = 0.5\left( \mathbf{C}_m(\mathbf{\xi}_i^r)+\mathbf{C}_m(\mathbf{\xi}_j^r) \right) \cdot \mathbf{f}_s(\mathbf{u}(\mathbf{\xi}_i^r),\mathbf{u}(\mathbf{\xi}_j^r)) \f$; that is, the 
    * matrix of REFERENCE two-point entropy conserving fluxes.
    */
    template <typename adtype>
    void assemble_volume_term_strong(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::array<std::vector<adtype>,nstate>           &soln_coeff,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>           &aux_soln_coeff,
        const unsigned int                                     poly_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper,
        OPERATOR::metric_operators<adtype,dim,2*dim>           &metric_oper,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>    &pde_physics,
        std::vector<adtype>                                    &local_rhs_int_cell);

    /// Strong form primary equation's boundary right-hand-side.
    template <typename adtype>
    void assemble_boundary_term_strong(
        typename dealii::DoFHandler<dim>::active_cell_iterator             current_cell,
        const unsigned int                                                 iface, 
        const dealii::types::global_dof_index                              current_cell_index,
        std::vector<bool>                                                  face_orientation,
        const std::array<std::vector<adtype>,nstate>                       &soln_coeff,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_coeff,
        const unsigned int                                                 boundary_id,
        const unsigned int                                                 poly_degree, 
        const real                                                         penalty,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>                &pde_physics,
        const NumericalFlux::NumericalFluxConvective<dim, nspecies, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        std::vector<adtype>                                                &local_rhs_cell);

    /// Strong form primary equation's facet right-hand-side.
    /**
    * \f[
    * \int_{\mathbf{\Gamma}_r}{\chi}_i(\mathbf{\xi}^r) \Big[ 
    * \hat{\mathbf{n}}^r\mathbf{C}_m^T \cdot \mathbf{f}^*_m - \hat{\mathbf{n}}^r \cdot \mathbf{\chi}(\mathbf{\xi}^r)\mathbf{\hat{f}}^r_m(t)^T
    * \Big]d \mathbf{\Gamma}_r
    * ,\:\forall i=1,\dots,N_p.
    * \f]
    */
    template <typename adtype>
    void assemble_face_term_strong(
        const unsigned int                                                 iface, 
        const unsigned int                                                 neighbor_iface, 
        const dealii::types::global_dof_index                              current_cell_index,
        const dealii::types::global_dof_index                              neighbor_cell_index,
        std::vector<bool>                                                  face_orientation_int,
        std::vector<bool>                                                  face_orientation_ext,
        const std::array<std::vector<adtype>,nstate>                       &soln_coeff_int,
        const std::array<std::vector<adtype>,nstate>                       &soln_coeff_ext,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_coeff_int,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &aux_soln_coeff_ext,
        const unsigned int                                                 poly_degree_int, 
        const unsigned int                                                 poly_degree_ext, 
        const real                                                         penalty,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
        Physics::PhysicsBase<dim, nspecies, nstate, adtype>                &pde_physics,
        const NumericalFlux::NumericalFluxConvective<dim, nspecies, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nspecies, nstate, adtype> &diss_num_flux,
        std::vector<adtype>                                                  &local_rhs_int_cell,
        std::vector<adtype>                                                  &local_rhs_ext_cell);

protected:
    /// Evaluate the integral over the cell volume
    void assemble_volume_term_explicit(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange);
    
    

    using DGBase<dim,nspecies,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian).
    template <typename adtype>
    void build_volume_metric_operators(
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        const std::vector<adtype>                    &metric_coeffs,
        OPERATOR::metric_operators<adtype,dim,2*dim> &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim> &mapping_basis,
        std::array<std::vector<adtype>,dim>          &mapping_support_points);
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). For double type.
    void build_volume_metric_operators(
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        const std::vector<double>                    &metric_coeffs,
        OPERATOR::metric_operators<double,dim,2*dim> &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim> &mapping_basis,
        std::array<std::vector<double>,dim>          &mapping_support_points)
    {
        build_volume_metric_operators<double>(poly_degree, grid_degree, metric_coeffs, metric_oper, mapping_basis, mapping_support_points);
    }
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). For codi_JacobianComputationType.
    void build_volume_metric_operators(
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        const std::vector<codi_JacobianComputationType>                    &metric_coeffs,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim> &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
        std::array<std::vector<codi_JacobianComputationType>,dim>          &mapping_support_points)
    {
        build_volume_metric_operators<codi_JacobianComputationType>(poly_degree, grid_degree, metric_coeffs, metric_oper, mapping_basis, mapping_support_points);
    }
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). For codi_HessianComputationType.
    void build_volume_metric_operators(
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        const std::vector<codi_HessianComputationType>                    &metric_coeffs,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim> &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>                      &mapping_basis,
        std::array<std::vector<codi_HessianComputationType>,dim>          &mapping_support_points)
    {
        build_volume_metric_operators<codi_HessianComputationType>(poly_degree, grid_degree, metric_coeffs, metric_oper, mapping_basis, mapping_support_points);
    }
    
}; // end of DGStrong class

/// Class containing functions pertaining to the entropy stable viscous discretization.
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nspecies, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nspecies, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EntropyStable_viscousBR2
{
    public:
        EntropyStable_viscousBR2(){}

    private:
    /// Projects tensor values at the volume quadrature to obtain a polynomial and interpolates that polynomial to the face quadratures.
    template <typename adtype>
    static void interpolate_to_face(
        std::vector<bool> face_orientation,
        const unsigned int iface,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &T_at_vol,
        OPERATOR::basis_functions<dim,2*dim> &flux_basis,
        const unsigned int n_face_quad_pts,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &T_at_face);

    /// Evaluates integral of a tensor at the face
    template <typename adtype>
    static void evaluate_face_integral(
        std::vector<bool> face_orientation,
        const unsigned int iface,
        const std::array<std::vector<adtype>,nstate> &sigma_dot_n_at_face,
        const std::vector<adtype> &JxW_face,
        OPERATOR::basis_functions<dim,2*dim> &soln_basis,
        const unsigned int n_dofs_cell,
        std::vector<adtype> &integral_val);

    /// Computes polynomial based on the values at the face
    template <typename adtype>
    static void compute_lift_polynomial(
        std::vector<bool> face_orientation,
        const unsigned int iface,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &phi_at_face,
        const std::vector<adtype> &JxW_face,
        const std::vector<adtype> &JxW_vol,
        OPERATOR::basis_functions<dim,2*dim> &flux_basis,
        const unsigned int n_face_quad_pts,
        const unsigned int n_vol_quad_pts,
        const bool is_interior_face,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &re_out_vol);

    /// Computes physical gradient of the entropy variable
    template <typename adtype>
    static void compute_physical_grad_entropy_var(
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        const unsigned int                                                 n_quad_pts,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       &entropy_var_phys_grad);

    /// Applies diffusion matrix entropy based (referred to as K in Jesse Chan's paper).
    template <typename adtype>
    static void apply_diffusion_matrix_entropy_based(
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_quads,
        const unsigned int                                                 n_quad_pts,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>          &pde_physics,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T_in,
        std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>         &T_out);

    
    /// Computes the volume integral of gradient of basis times diffusion matrix times input tensor
    template <typename adtype>
    static void compute_gradbasis_diffusionmatrix_times_input_vol_integral(
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_quads,
        const unsigned int                                                 n_quad_pts,
        const unsigned int                                                 n_dofs_cell,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        const std::vector<double>                                          &weight_vect,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>         &pde_physics,
        const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T_input,
        std::vector<adtype>                                                &integral_val);
    
    public:
    /// Assemble BR2 volume term.
    template <typename adtype>
    static void assemble_volume_term_entropystable_br2(
        const unsigned int                                                 poly_degree,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_q,
        const unsigned int                                                  n_quad_pts,
        const unsigned int                                                  n_dofs_cell,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>         &pde_physics,
        const std::vector<double>                                          &weight_vect,
        std::vector<adtype>                                                &vol_term);

    /// Assemble BR2 face term.
    template <typename adtype>
    static void assemble_face_term_entropystable_br2(
        std::vector<bool>                                                  face_orientation_int,
        std::vector<bool>                                                  face_orientation_ext,
        const unsigned int                                                 iface_int,
        const unsigned int                                                 iface_ext,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_ext,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_ext,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_int,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_ext,
        const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal_int,
        const std::vector<adtype>                                          &JxW_face,
        const unsigned int                                                  poly_degree_int,
        const unsigned int                                                  poly_degree_ext,
        const unsigned int                                                  n_vol_quad_pts_int,
        const unsigned int                                                  n_vol_quad_pts_ext,
        const unsigned int                                                  n_dofs_cell_int,
        const unsigned int                                                  n_dofs_cell_ext,
        const unsigned int                                                  n_face_quad_pts,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>          &pde_physics,
        const std::vector<double>                                          &vol_quad_weights_int,
        const std::vector<double>                                          &vol_quad_weights_ext,
        std::vector<adtype>                                                &face_term_int,
        std::vector<adtype>                                                &face_term_ext);

    /// Assemble BR2 boundary term.
    template <typename adtype>
    static void assemble_boundary_term_entropystable_br2(
        std::vector<bool>                                                  face_orientation,
        const unsigned int                                                 iface,
        const unsigned int                                                 boundary_id,
        const unsigned int                                                 poly_degree,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_quads,
        const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_quads,
        const unsigned int                                                  n_vol_quad_pts,
        const unsigned int                                                  n_face_quad_pts,
        const unsigned int                                                  n_dofs_cell,
        OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
        const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal,
        const std::vector<adtype>                                          &JxW_face,
        const Physics::PhysicsBase<dim, nspecies, nstate, adtype>          &pde_physics,
        const std::vector<double>                                          &vol_quad_weights,
        std::vector<adtype>                                                &boundary_term);
};

} // PHiLiP namespace

#endif
