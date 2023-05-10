#ifndef __IMPLICIT_SHOCKTRACKING_FUNCTIONAL_H__
#define __IMPLICIT_SHOCKTRACKING_FUNCTIONAL_H__

#include "functional.h"

namespace PHiLiP {

/// Class to compute the objective function of fine residual used for optimization based mesh adaptation. \f[ \mathcal{F} = \frac{1}{2} R^T R \f].
template <int dim, int nstate, typename real>
class ImplicitShockTrackingFunctional : public Functional<dim, nstate, real>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    ImplicitShockTrackingFunctional( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = false,
        const bool _use_coarse_residual = false);

    /// Destructor
    ~ImplicitShockTrackingFunctional(){}

    /// Computes \f[ out_vector = d2IdWdW*in_vector \f]. 
    void d2IdWdW_vmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdWdX*in_vector \f]. 
    void d2IdWdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override; 
    /// Computes \f[ out_vector = d2IdWdX^T*in_vector \f]. 
    void d2IdWdX_Tvmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdXdX*in_vector \f]. 
    void d2IdXdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override;

    /// Evaluates \f[ \mathcal{F} = \frac{1}{2} \sum_k \eta_k^2 \f] and derivatives, if needed.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    /// Stores true if coarse residual is used in the objective function.
    const bool use_coarse_residual;

    /// Extracts all matrices possible for various combinations of polynomial degrees.
    void extract_interpolation_matrices(dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp);
    
    /// Returns cellwise dof indices. Used to store cellwise dof indices of higher poly order grid to form interpolation matrix and cto compute matrix-vector products.
    std::vector<std::vector<dealii::types::global_dof_index>> get_cellwise_dof_indices();

    /// Evaluates objective function and stores adjoint and residual.
    real evaluate_objective_function();

    /// Computes common vectors and matrices (R_u, R_u_transpose, adjoint*d2R) required for dIdW, dIdX and d2I.
    void compute_common_vectors_and_matrices();

    /// Computes interpolation matrix.
    /** Assumes the polynomial order remains constant throughout the optimization algorithm.
     *  Also assumes that all cells have the same polynomial degree.
     */
    void compute_interpolation_matrix();
    
    /// Computes projection matrix of size i x j (i.e. projects from fe_j to fe_i).
    void get_projection_matrix(
        const dealii::FESystem<dim,dim> &fe_i, 
        const dealii::FESystem<dim,dim> &fe_j, 
        dealii::FullMatrix<real> &projection_matrix);

    /// Stores dIdW
    void store_dIdW();

    /// Stores dIdX
    void store_dIdX();

    /// Stores \f[R_u \f] on fine space. 
    MatrixType R_u;
    
    /// Stores \f[R_x \f] on fine space. 
    MatrixType R_x;

    /// Stores \f[R^T R_ux \f] on fine space. 
    MatrixType R_times_Rux;

    /// Stores \f[R^T R_uu \f] on fine space. 
    MatrixType R_times_Ruu;

    /// Stores \f[R^T R_xx \f] on fine space. 
    MatrixType R_times_Rxx;
 
    /// Residual used to evaluate objective function.
    VectorType residual_fine;

    /// Stores vector on coarse space to copy parallel partitioning later.
    VectorType vector_coarse;
    
    /// Stores vector on fine space (p+1) to copy parallel partitioning later.
    VectorType vector_fine;
    
    /// Stores vector of volume nodes to copy parallel partitioning later.
    VectorType vector_vol_nodes;
    
    /// Stores the weight of mesh distortion term added to the objective function.
    const real mesh_weight;

    /// Stores initial volume nodes for implementing mesh weight.
    const VectorType initial_vol_nodes;
    
    /// Stores the coarse poly degree.
    const unsigned int coarse_poly_degree;
    
    /// Stores the fine poly degree.
    const unsigned int fine_poly_degree;

    /// Functional used to evaluate cell distortion.
    std::unique_ptr< Functional<dim, nstate, real> > cell_distortion_functional;
    
public:
    /// Stores global dof indices of the fine mesh.
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dofs_fine;

    /// Stores interpolation matrix \f[ I_h \f] to interpolate onto fine space. Used to compute \f[ U_h^H = I_h u_H \f]. 
    MatrixType interpolation_matrix;    
};

} // namespace PHiLiP

#endif
