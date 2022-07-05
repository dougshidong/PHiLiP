#ifndef __STRONG_DISCONTINUOUSGALERKIN_H__
#define __STRONG_DISCONTINUOUSGALERKIN_H__

#include "dg.h"

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

    /// Destructor
    ~DGStrong();

    ///Allocates the auxiliary equations' variables and RHS.
    void allocate_auxiliary_equation ();

    ///Assembles the auxiliary equations' residuals and solves for the auxiliary variables.
    void assemble_auxiliary_residual ();
private:

    ///Assembles the auxiliary equations' cell residuals.
    template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
    void assemble_cell_auxiliary_residual (
        const DoFCellAccessorType1 &current_cell,
        const DoFCellAccessorType2 &current_metric_cell,
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rhs);

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
    /** \f[
    * \int_{\mathbf{\Gamma}_r} \chi_i \left( \hat{\mathbf{n}}^r\mathbf{C}_m(\mathbf{\xi}^r)^T\right) \cdot \left[ u^*  
    * - u\right]d\mathbf{\Gamma}_r,\:\forall i=1,\dots,N_p.
    * \f]
    */
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
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>         &metric_oper_ext,
        dealii::Vector<real>                               &local_rhs_int_cell,
        dealii::Vector<real>                               &local_rhs_ext_cell);


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
        const unsigned int iface, 
        const unsigned int neighbor_iface,
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


    using DGBase<dim,real,MeshType>::all_parameters; ///< Pointer to all parameters
    using DGBase<dim,real,MeshType>::mpi_communicator; ///< MPI communicator
    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // end of DGStrong class

} // PHiLiP namespace

#endif

