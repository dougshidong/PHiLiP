#ifndef __STRONG_DISCONTINUOUSGALERKIN_H__
#define __STRONG_DISCONTINUOUSGALERKIN_H__

#include "dg_base_state.hpp"

namespace PHiLiP {

/// DGStrong class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGStrong: public DGBaseState<dim, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGBaseState<dim,nstate,real,MeshType>::Triangulation;

public:
    /// Constructor
    DGStrong(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// Assembles the auxiliary equations' residuals and solves for the auxiliary variables.
    /** For information regarding auxiliary vs. primary quations, see 
     *  Quaegebeur, Nadarajah, Navah and Zwanenburg 2019: Stability of Energy Stable Flux 
     *                Reconstruction for the Diffusion Problem Using Compact Numerical Fluxes
     */
    void assemble_auxiliary_residual ();

    /// Allocate the dual vector for optimization.
    void allocate_dual_vector ();

private:
    /// Assembles the auxiliary equations' cell residuals.
    template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
    void assemble_cell_auxiliary_residual (
        const DoFCellAccessorType1 &current_cell,
        const DoFCellAccessorType2 &current_metric_cell,
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rhs);

protected:
    /// Builds the necessary operators and assembles volume residual for either primary or auxiliary.
    void assemble_volume_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume*/,
        dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume_lagrange*/,
        const dealii::FESystem<dim,dim>                        &/*current_fe_ref*/,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/);

    /// Builds the necessary operators and assembles boundary residual for either primary or auxiliary.
    void assemble_boundary_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
        const dealii::types::global_dof_index                  current_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     boundary_id,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_int*/,
        const dealii::FESystem<dim,dim>                        &/*current_fe_ref*/,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool /*compute_dRdW*/, const bool /*compute_dRdX*/, const bool /*compute_d2R*/);

    /// Builds the necessary operators and assembles face residual.
    void assemble_face_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
        typename dealii::DoFHandler<dim>::active_cell_iterator /*neighbor_cell*/,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     poly_degree_int,
        const unsigned int                                     poly_degree_ext,
        const unsigned int                                     grid_degree_int,
        const unsigned int                                     grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_int*/,
        dealii::hp::FEFaceValues<dim,dim>                      &/*fe_values_collection_face_ext*/,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    /// Builds the necessary operators and assembles subface residual.
    /** Not verified
    */
    void assemble_subface_term_and_build_operators(
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
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                   &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim>                   &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FESubfaceValues<dim,dim>                   &/*fe_values_collection_subface*/,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

public:
    ///Evaluate the volume RHS for the auxiliary equation.
    /** \f[
    * \int_{\mathbf{\Omega}_r} \chi_i(\mathbf{\xi}^r) \left( \nabla^r(u) \right)\mathbf{C}_m(\mathbf{\xi}^r) d\mathbf{\Omega}_r,\:\forall i=1,\dots,N_p.
    * \f]
    */
    void assemble_volume_term_auxiliary_equation(
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const unsigned int                                 poly_degree,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS);

protected:
    /// Evaluate the boundary RHS for the auxiliary equation.
    void assemble_boundary_term_auxiliary_equation(
        const unsigned int                                 iface,
        const dealii::types::global_dof_index              current_cell_index,
        const unsigned int                                 poly_degree,
        const unsigned int                                 boundary_id,
        const std::vector<dealii::types::global_dof_index> &dofs_indices,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        std::vector<dealii::Tensor<1,dim,real>>            &local_auxiliary_RHS);

public:
    /// Evaluate the facet RHS for the auxiliary equation.
    /** \f[
    * \int_{\mathbf{\Gamma}_r} \chi_i \left( \hat{\mathbf{n}}^r\mathbf{C}_m(\mathbf{\xi}^r)^T\right) \cdot \left[ u^*  
    * - u\right]d\mathbf{\Gamma}_r,\:\forall i=1,\dots,N_p.
    * \f]
    */
    void assemble_face_term_auxiliary_equation(
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
    void assemble_volume_term_strong(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index              current_cell_index,
        const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
        const unsigned int                                 poly_degree,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim>         &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim>       &soln_basis_projection_oper,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        dealii::Vector<real>                               &local_rhs_int_cell);

    /// Strong form primary equation's boundary right-hand-side.
    void assemble_boundary_term_strong(
        const unsigned int                                 iface, 
        const dealii::types::global_dof_index              current_cell_index,
        const unsigned int                                 boundary_id,
        const unsigned int                                 poly_degree, 
        const real                                         penalty,
        const std::vector<dealii::types::global_dof_index> &dof_indices,
        OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
        OPERATOR::basis_functions<dim,2*dim>               &flux_basis,
        OPERATOR::vol_projection_operator<dim,2*dim>       &soln_basis_projection_oper,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper,
        dealii::Vector<real>                               &local_rhs_cell);

    /// Strong form primary equation's facet right-hand-side.
    /**
    * \f[
    * \int_{\mathbf{\Gamma}_r}{\chi}_i(\mathbf{\xi}^r) \Big[ 
    * \hat{\mathbf{n}}^r\mathbf{C}_m^T \cdot \mathbf{f}^*_m - \hat{\mathbf{n}}^r \cdot \mathbf{\chi}(\mathbf{\xi}^r)\mathbf{\hat{f}}^r_m(t)^T
    * \Big]d \mathbf{\Gamma}_r
    * ,\:\forall i=1,\dots,N_p.
    * \f]
    */
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
        OPERATOR::vol_projection_operator<dim,2*dim>       &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim>       &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_ext,
        dealii::Vector<real>                               &local_rhs_int_cell,
        dealii::Vector<real>                               &local_rhs_ext_cell);

protected:
    /// Evaluate the integral over the cell volume and the specified derivatives.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. */
    void assemble_volume_term_derivatives(
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
    
    /// Assemble boundary term derivatives
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
    
}; // end of DGStrong class

} // PHiLiP namespace

#endif
