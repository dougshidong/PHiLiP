#ifndef __WEAK_DISCONTINUOUSGALERKIN_H__
#define __WEAK_DISCONTINUOUSGALERKIN_H__

#include "dg.h"

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

    ~DGWeak(); ///< Destructor.

private:

    /// Builds the necessary fe values and assembles volume residual.
    void assemble_volume_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                        &current_fe_ref,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &/*local_auxiliary_RHS*/,
        const bool                                             /*compute_Auxiliary_RHS*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Builds the necessary fe values and assembles boundary residual.
    void assemble_boundary_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     boundary_id,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                        &current_fe_ref,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &/*local_auxiliary_RHS*/,
        const bool                                             /*compute_Auxiliary_RHS*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Builds the necessary fe values and assembles face residual.
    void assemble_face_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     /*poly_degree_int*/,
        const unsigned int                                     /*poly_degree_ext*/,
        const unsigned int                                     /*grid_degree_int*/,
        const unsigned int                                     /*grid_degree_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_int*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_ext*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_ext,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &/*current_cell_rhs_aux*/,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &/*rhs_aux*/,
        const bool                                             /*compute_Auxiliary_RHS*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Builds the necessary fe values and assembles subface residual.
    void assemble_subface_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const unsigned int                                     neighbor_i_subface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     /*poly_degree_int*/,
        const unsigned int                                     /*poly_degree_ext*/,
        const unsigned int                                     /*grid_degree_int*/,
        const unsigned int                                     /*grid_degree_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*soln_basis_ext*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim>                   &/*flux_basis_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_int*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_ext*/,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FESubfaceValues<dim,dim>                   &fe_values_collection_subface,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &/*current_cell_rhs_aux*/,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &/*rhs_aux*/,
        const bool                                             /*compute_Auxiliary_RHS*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    ///Allocates the auxiliary equations' variables and RHS.
    void allocate_auxiliary_equation ();

    ///Assembles the auxiliary equations' residuals and solves for the auxiliary variables.
    void assemble_auxiliary_residual ();
    ///Evaluate the volume RHS for the auxiliary equation.
    void assemble_volume_term_auxiliary_equation(
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const unsigned int                                 poly_degree,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS);
    ///Evaluate the boundary RHS for the auxiliary equation.
    void assemble_boundary_term_auxiliary_equation(
        const unsigned int                                 iface,
        const dealii::types::global_dof_index              current_cell_index,
        const unsigned int                                 poly_degree,
        const unsigned int                                 boundary_id,
        const std::vector<dealii::types::global_dof_index> &dofs_indices,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS);
    ///Evaluate the facet RHS for the auxiliary equation.
    void assemble_face_term_auxiliary(
        const unsigned int                                 iface, 
        const unsigned int                                 neighbor_iface,
        const dealii::types::global_dof_index              current_cell_index,
        const dealii::types::global_dof_index              neighbor_cell_index,
        const unsigned int                                 poly_degree_int, 
        const unsigned int                                 poly_degree_ext,
        const std::vector<dealii::types::global_dof_index> &dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_ext,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_int,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS_int,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS_ext);
    ///Strong form primary equation's volume right-hand-side.
    void assemble_volume_term_strong(
        const dealii::types::global_dof_index              current_cell_index,
        const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
        const unsigned int                                 poly_degree,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        dealii::Vector<real>                               &local_rhs_int_cell);
    ///Strong form primary equation's boundary right-hand-side.
    void assemble_boundary_term_strong(
        const unsigned int                                 iface, 
        const dealii::types::global_dof_index              current_cell_index,
        const unsigned int                                 boundary_id,
        const unsigned int                                 poly_degree, 
        const real                                         penalty,
        const std::vector<dealii::types::global_dof_index> &dof_indices,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        dealii::Vector<real>                               &local_rhs_cell);
    ///Strong form primary equation's facet right-hand-side.
    void assemble_face_term_strong(
        const unsigned int                                 iface, 
        const unsigned int                                 neighbor_iface, 
        const dealii::types::global_dof_index              current_cell_index,
        const dealii::types::global_dof_index              neighbor_cell_index,
        const unsigned int                                 poly_degree_int, 
        const unsigned int                                 poly_degree_ext, 
        const real                                         penalty,
        const std::vector<dealii::types::global_dof_index> &dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis_ext,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_ext,
        dealii::Vector<real>                               &local_rhs_int_cell,
        dealii::Vector<real>                               &local_rhs_ext_cell);

    /// Main function responsible for evaluating the integral over the cell volume and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename real2>
    void assemble_volume_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<real2> &soln_coeff,
        const std::vector<real2> &coords_coeff,
        const std::vector<real> &local_dual,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::FESystem<dim,dim> &fe_metric,
        const dealii::Quadrature<dim> &quadrature,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        std::vector<real2> &rhs,
        real2 &dual_dot_residual,
        const bool compute_metric_derivatives,
        const dealii::FEValues<dim,dim> &fe_values_vol);

    /// Main function responsible for evaluating the boundary integral and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename adtype>
    void assemble_boundary_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector< adtype > &soln_coeff,
        const std::vector< adtype > &coords_coeff,
        const std::vector< real > &local_dual,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const Physics::PhysicsBase<dim, nstate, adtype> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::FESystem<dim,dim> &fe_metric,
        const dealii::Quadrature<dim-1> &quadrature,
        std::vector<adtype> &rhs,
        adtype &dual_dot_residual,
        const bool compute_metric_derivatives);

    /// Main function responsible for evaluating the internal face integral and the specified derivatives.
    /** This function templates the solution and metric coefficients in order to possible AD the residual.
     */
    template <typename real2>
    void assemble_face_term(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::vector< real2 > &soln_coeff_int,
        const std::vector< real2 > &soln_coeff_ext,
        const std::vector< real2 > &coords_coeff_int,
        const std::vector< real2 > &coords_coeff_ext,
        const std::vector< double > &dual_int,
        const std::vector< double > &dual_ext,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::FESystem<dim,dim> &fe_metric,
        const dealii::Quadrature<dim-1> &face_quadrature,
        std::vector<real2> &rhs_int,
        std::vector<real2> &rhs_ext,
        real2 &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

private:

    /// Preparation of CoDiPack taping for volume integral, and derivative evaluation.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. 
     *  Uses CoDiPack to automatically differentiate the functions.
     */
    template <typename real2>
    void assemble_volume_codi_taped_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &fe_values_vol,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::Quadrature<dim> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const dealii::FEValues<dim,dim> &fe_values_lagrange,
        const Physics::PhysicsBase<dim, nstate, real2> &physics,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Preparation of CoDiPack taping for boundary integral, and derivative evaluation.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. 
     *  Uses CoDiPack to automatically differentiate the functions.
     */
    template <typename adtype>
    void assemble_boundary_codi_taped_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::Quadrature<dim-1> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        const Physics::PhysicsBase<dim, nstate, adtype> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
        dealii::Vector<real> &local_rhs_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Preparation of CoDiPack taping for internal cell faces integrals, and derivative evaluation.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. 
     *  Uses CoDiPack to automatically differentiate the functions.
     *  This adds the contribution to both cell's residual and effectively
     *  computes 4 block contributions to dRdX blocks.
     */
    template <typename adtype>
    void assemble_face_codi_taped_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
        const Physics::PhysicsBase<dim, nstate, adtype> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
        dealii::Vector<real>          &local_rhs_int_cell,
        dealii::Vector<real>          &local_rhs_ext_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);


private:

    /// Evaluate the integral over the cell volume and the specified derivatives.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. */
    virtual void assemble_volume_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &,//fe_values_vol,
        const dealii::FESystem<dim,dim> &fe,
        const dealii::Quadrature<dim> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);


    /// Evaluate the integral over the cell edges that are on domain boundaries and the specified derivatives.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. */
    void assemble_boundary_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe,
        const dealii::Quadrature<dim-1> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);


    /// Evaluate the integral over the internal cell edges and its specified derivatives.
    /** Compute both the right-hand side and the block of the Jacobian.
     *  This adds the contribution to both cell's residual and effectively
     *  computes 4 block contributions to dRdX blocks. */
    void assemble_face_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
        dealii::Vector<real>          &local_rhs_int_cell,
        dealii::Vector<real>          &local_rhs_ext_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

private: 

    /// Evaluate the integral over the cell volume.
    /** Compute the right-hand side only. */
    void assemble_volume_residual(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &fe_values_vol,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::Quadrature<dim> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const dealii::FEValues<dim,dim> &fe_values_lagrange,
        const Physics::PhysicsBase<dim, nstate, real> &physics,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Evaluate the integral over the boundary.
    /** Compute the right-hand side only. */
    void assemble_boundary_residual(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_soln,
        const dealii::Quadrature<dim-1> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        const Physics::PhysicsBase<dim, nstate, real> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, real> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, real> &diss_num_flux,
        dealii::Vector<real> &local_rhs_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Evaluate the integral over the internal face.
    /** Compute the right-hand side only. */
    void assemble_face_residual(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
        const Physics::PhysicsBase<dim, nstate, real> &physics,
        const NumericalFlux::NumericalFluxConvective<dim, nstate, real> &conv_num_flux,
        const NumericalFlux::NumericalFluxDissipative<dim, nstate, real> &diss_num_flux,
        dealii::Vector<real>          &local_rhs_int_cell,
        dealii::Vector<real>          &local_rhs_ext_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);


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
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_explicit(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const unsigned int boundary_id,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    void assemble_face_term_explicit(
        const unsigned int /*iface*/, 
        const unsigned int /*neighbor_iface*/,
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const unsigned int poly_degree, 
        const unsigned int grid_degree,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

    using DGBase<dim,real,MeshType>::mpi_communicator; ///< MPI communicator
    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // end of DGWeak class

} // PHiLiP namespace

#endif

