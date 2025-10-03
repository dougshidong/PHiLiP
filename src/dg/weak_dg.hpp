#ifndef __WEAK_DISCONTINUOUSGALERKIN_H__
#define __WEAK_DISCONTINUOUSGALERKIN_H__

#include "dg_base_state.hpp"
#include "solution/local_solution.hpp"

namespace PHiLiP {

/// DGWeak class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGWeak : public DGBaseState<dim, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGBaseState<dim,nstate,real,MeshType>::Triangulation;
public:
    /// Constructor.
    DGWeak(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

private:

    /// Assembles the auxiliary equations' residuals and solves for the auxiliary variables.
    void assemble_auxiliary_residual (const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/);

    /// Allocate the dual vector for optimization.
    void allocate_dual_vector (const bool compute_d2R);

    /// Main function responsible for evaluating the integral over the cell volume and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename real2>
    void assemble_volume_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const LocalSolution<real2, dim, nstate> &local_solution,
        const LocalSolution<real2, dim, dim> &local_metric,
        const std::vector<real> &local_dual,
        const dealii::Quadrature<dim> &quadrature,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        std::vector<real2> &rhs, real2 &dual_dot_residual,
        const bool compute_metric_derivatives,
        const dealii::FEValues<dim,dim> &fe_values_vol);

    /// Main function responsible for evaluating the boundary integral and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename real2>
    void assemble_boundary_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const LocalSolution<real2, dim, nstate> &local_solution,
        const LocalSolution<real2, dim, dim> &local_metric,
        const std::vector< real > &local_dual,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::Quadrature<dim-1> &quadrature,
        std::vector<real2> &rhs,
        real2 &dual_dot_residual,
        const bool compute_metric_derivatives);

    /// Main function responsible for evaluating the internal face integral and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename real2>
    void assemble_face_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const LocalSolution<real2, dim, nstate> &soln_int,
        const LocalSolution<real2, dim, nstate> &soln_ext,
        const LocalSolution<real2, dim, dim> &metric_int,
        const LocalSolution<real2, dim, dim> &metric_ext,
        const std::vector< double > &dual_int,
        const std::vector< double > &dual_ext,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
        const real penalty,
        const dealii::Quadrature<dim-1> &face_quadrature,
        std::vector<real2> &rhs_int,
        std::vector<real2> &rhs_ext,
        real2 &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);
    
    /// Calls the function to assemble volume residual.
    template <typename adtype>
    void assemble_volume_term_and_build_operators_ad_templated(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<adtype> &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<adtype>>        &/*aux_soln_coeffs*/,
        const std::vector<adtype> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        const Physics::PhysicsBase<dim, nstate, adtype> &physics,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis*/,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim>           &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim>           &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<adtype,dim,2*dim>           &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &/*mapping_basis*/,
        std::array<std::vector<adtype>,dim>                    &/*mapping_support_points*/,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<adtype> &rhs, 
        dealii::Tensor<1,dim,std::vector<adtype>>              &/*local_auxiliary_RHS*/,
        const bool                                             /*compute_auxiliary_right_hand_side*/,
        adtype &dual_dot_residual);
    
    /// Calls the function to assemble volume residual. For double type.
    void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<double> &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<double>>        &aux_soln_coeffs,
        const std::vector<double> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<double,dim,2*dim>           &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<double>,dim>                    &mapping_support_points,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<double> &rhs, 
        dealii::Tensor<1,dim,std::vector<double>>              &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        double &dual_dot_residual) override
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
                *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
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
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_JacobianComputationType> &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>> &aux_soln_coeffs,
        const std::vector<codi_JacobianComputationType> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_JacobianComputationType,dim,2*dim>           &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<codi_JacobianComputationType>,dim> &mapping_support_points,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<codi_JacobianComputationType> &rhs, 
        dealii::Tensor<1,dim,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        codi_JacobianComputationType &dual_dot_residual) override
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
                *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
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
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_HessianComputationType> &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>> &aux_soln_coeffs,
        const std::vector<codi_HessianComputationType> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<codi_HessianComputationType,dim,2*dim> &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<codi_HessianComputationType>,dim> &mapping_support_points,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<codi_HessianComputationType> &rhs, 
        dealii::Tensor<1,dim,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        codi_HessianComputationType &dual_dot_residual) override
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
                *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
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
    
    /// Calls the function to assemble boundary residual.
    template <typename adtype>
    void assemble_boundary_term_and_build_operators_ad_templated(
        typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
        const dealii::types::global_dof_index                              current_cell_index,
        const std::vector<adtype>                                          &soln_coeffs,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeffs*/,
        const std::vector<adtype>                                          &metric_coeffs,
        const std::vector<real>                                            &local_dual,
        const unsigned int                                                 face_number,
        const unsigned int                                                 boundary_id,
        const Physics::PhysicsBase<dim, nstate, adtype>                    &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
        const unsigned int                                                 /*poly_degree*/,
        const unsigned int                                                 /*grid_degree*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis*/,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper_int*/,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &/*mapping_basis*/,
        std::array<std::vector<adtype>,dim>                                &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                                    &fe_soln,
        const real                                                         penalty,
        std::vector<adtype>                                                &rhs,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &/*local_auxiliary_RHS*/,
        const bool                                                         /*compute_auxiliary_right_hand_side*/,
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double),
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad),
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad_fad),
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
        const std::vector<adtype>                                          &soln_coeff_int,
        const std::vector<adtype>                                          &soln_coeff_ext,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeff_int*/,
        const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeff_ext*/,
        const std::vector<adtype>                                          &metric_coeff_int,
        const std::vector<adtype>                                          &metric_coeff_ext,
        const std::vector< double >                                        &dual_int,
        const std::vector< double >                                        &dual_ext,
        const unsigned int                                                 /*poly_degree_int*/,
        const unsigned int                                                 /*poly_degree_ext*/,
        const unsigned int                                                 /*grid_degree_int*/,
        const unsigned int                                                 /*grid_degree_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*soln_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*soln_basis_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis_ext*/,
        OPERATOR::local_basis_stiffness<dim,2*dim>                         &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &/*metric_oper_int*/,
        OPERATOR::metric_operators<adtype,dim,2*dim>                       &/*metric_oper_ext*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>                       &/*mapping_basis*/,
        std::array<std::vector<adtype>,dim>                                &/*mapping_support_points*/,
        const Physics::PhysicsBase<dim, nstate, adtype>                    &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                               &fe_values_collection_subface,
        const dealii::FESystem<dim,dim>                                    &fe_int,
        const dealii::FESystem<dim,dim>                                    &fe_ext,
        const real                                                         penalty,
        std::vector<adtype>                                                &rhs_int,
        std::vector<adtype>                                                &rhs_ext,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &/*aux_rhs_int*/,
        dealii::Tensor<1,dim,std::vector<adtype>>                          &/*aux_rhs_ext*/,
        const bool                                                         /*compute_auxiliary_right_hand_side*/,
        adtype                                                             &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
        const bool                                                         is_a_subface,
        const unsigned int                                                 neighbor_i_subface);
    
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
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double),
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
            compute_dRdW, compute_dRdX, compute_d2R,
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
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad),
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
            compute_dRdW, compute_dRdX, compute_d2R,
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
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
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
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad_fad),
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
            compute_dRdW, compute_dRdX, compute_d2R,
            is_a_subface,
            neighbor_i_subface);
    }

private: 
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
    

    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). Does nothing for weak DG.
    template <typename adtype>
    void build_volume_metric_operators(
        const unsigned int /*poly_degree*/,
        const unsigned int /*grid_degree*/,
        const std::vector<adtype>                    &/*metric_coeffs*/,
        OPERATOR::metric_operators<adtype,dim,2*dim> &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim> &/*mapping_basis*/,
        std::array<std::vector<adtype>,dim>          &/*mapping_support_points*/)
    {
        //Do Nothing.
    }
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). Does nothing for weak DG. For double type.
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
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). Does nothing for weak DG. For codi_JacobianComputationType.
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
    /// Builds volume metric operators (metric cofactor and determinant of metric Jacobian). Does nothing for weak DG. For codi_HessianComputationType.
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

}; // end of DGWeak class

} // PHiLiP namespace

#endif
