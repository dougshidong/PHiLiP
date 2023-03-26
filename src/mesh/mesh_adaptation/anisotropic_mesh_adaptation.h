#ifndef __ANISOTROPIC_MESH_ADAPTATION__
#define __ANISOTROPIC_MESH_ADAPTATION__

#include "dg/dg.h"
#include "functional/functional.h"

namespace PHiLiP {

/** Performs anisotropic mesh adaptation with an optimal metric for P1 solution.
 *  Implements the optimal metric field derived from continuous optimization framework. 
 *  See papers from INRIA for further details: 
 *  Feature based : Loseille, A. and Alauzet, F. "Continuous mesh framework part I: well-posed continuous interpolation error.", 2011.
 *  Goal oriented: Loseille, A., Dervieux, A., and Alauzet, F. "Fully anisotropic goal-oriented mesh adaptation for 3d steady Euler equations.", 2010.
 * @note The goal oriented approach is currently implemented for convection dominated flows.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class AnisotropicMeshAdaptation {
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.

public:
    /// Constructor
    AnisotropicMeshAdaptation(
        std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, 
        const real _norm_Lp,
        const real _complexity,
        const bool _use_goal_oriented_approach = false);

    /// Destructor
    ~AnisotropicMeshAdaptation(){};

    /// Function which adapts mesh and loads in new mesh.
    void adapt_mesh();

private:
    
    /// Returns positive tensor from an input tensor by taking absolute of eigenvalues.
    dealii::Tensor<2, dim, real> get_positive_definite_tensor(const dealii::Tensor<2, dim, real> &input_tensor) const;

    /// Computes optimal metric depending on goal oriented or feature based approach. 
    void compute_cellwise_optimal_metric();


    /// Computes hessian using the input coefficients, which can be a solution sensor or (for goal oriented approach) convective flux.
    /** This function is called by compute_optimal_metric(). 
     */
    void compute_abs_hessian();

    /// Initializes cellwise metric and hessian to zero tensors.
    void initialize_cellwise_metric_and_hessians();

    /// Computes feature based hessian (i.e. hessian of the solution).
    void compute_feature_based_hessian();

    /// Computes pseudo Hessian for the goal oriented approach.
    void compute_goal_oriented_hessian();

    /// Change the polynomial order and interpolate solution. 
    void change_p_degree_and_interpolate_solution(const unsigned int poly_degree);

    /// Reconstructs p2 solution after interpolation.
    /** Currenlty done using 1 linear solve of the implicit system.
     */
    void reconstruct_p2_solution();

    /// Returns quadrature number of a point which is closest to the reference cell's center.
    unsigned int get_iquad_near_cellcenter(const dealii::Quadrature<dim> &volume_quadrature) const;

    /// Returns flux and source term coeffs by evaluating flux at support points of fe. 
    void get_flux_and_source_at_support_pts(
            std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> &flux_at_support_pts, 
            std::vector<std::array< real, nstate>> &source_at_support_pts,
            const dealii::FEValues<dim,dim> &fe_values_input,
            const std::vector<dealii::types::global_dof_index> &dof_indices,
            typename dealii::DoFHandler<dim>::active_cell_iterator cell) const;

    /// Shared pointer to DGBase.
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    ///Flag to use goal oriented approach. It is set to false by default.
    const bool use_goal_oriented_approach;
    
    /// Stores hessian in each cell
    std::vector<dealii::Tensor<2, dim, real>> cellwise_hessian;
    
    /// Lp Norm w.r.t. which the anlytical optimization is done.
    const real normLp;

    /// Analogue of number of vertices/elements in continuous mesh framework.
    const real complexity;
    
    /// Alias for MPI_COMM_WORLD
    MPI_Comm mpi_communicator;
    
    /// std::cout only on processor #0.
    dealii::ConditionalOStream pcout;

    /// Processor# of current processor.
    int mpi_rank;

    /// Total no. of processors
    int n_mpi;

    /// Stores initial polynomial degree
    const unsigned int initial_poly_degree;

    /// Contains the physics of the PDE with real type
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, real > > pde_physics_double;
    
    /// Functional to evaluate the adjoint for goal oriented anisotropic meshing.
    std::shared_ptr< Functional<dim, nstate, real, MeshType> > functional;

    /// Stores optimal metric in each cell
    std::vector<dealii::Tensor<2, dim, real>> cellwise_optimal_metric;
    
    /// Stores max dofs per cell to initialize dof_indices.
    const unsigned int max_dofs_per_cell;
};

} // PHiLiP namepsace

#endif
