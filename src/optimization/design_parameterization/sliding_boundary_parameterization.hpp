#ifndef __DESIGN_PARAMETERIZATION_INNERSLIDING_H__
#define __DESIGN_PARAMETERIZATION_INNERSLIDING_H__

#include "base_parameterization.hpp"

namespace PHiLiP {

/// Design parameterization allowing boundary nodes to slide.
template<int dim>
class SlidingBoundaryParameterization : public BaseParameterization<dim> {
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    SlidingBoundaryParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid); 
    
    /// Destructor
    ~SlidingBoundaryParameterization() {};
    
    /// Initializes design variables with innersliding and sliding  volume nodes and set locally owned and ghost indices. Overrides the virtual function in base class.
    void initialize_design_variables(VectorType &design_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. innersliding and sliding volume nodes. Overrides the virtual function in base class.
    /** dXv_dXp is a rectangular matrix of dimension n_vol_nodes x n_innersliding_plus_boundary_nodes.
     */
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    /// Checks if the design variables have changed and updates volume nodes of high order grid. 
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    /// Returns the number of design variables (i.e. no. of innersliding and sliding volume nodes).
    unsigned int get_number_of_design_variables() const override;
    
    /// Checks if the updated design variable doesn't distort the mesh (which is possible when backtracking with high initial step length). Returns 0 if everything is good.
    int is_design_variable_valid(const MatrixType &dXv_dXp, const VectorType &design_var) const override;
    
    /// Computes innersliding_vol_range IndexSet on each processor, along with n_innersliding_nodes and innersliding_vol_index_to_vol_index. 
    /** Called by the constructor of the class. These variables are expected to be constant throughtout the optimization for now (as no new nodes are being added).
     */
    void compute_innersliding_vol_index_to_vol_index();

private:
    /// Current design variable. Stored to prevent recomputing if it is unchanged. 
    VectorType current_design_var;
    /// No. of innersliding volume nodes.
    unsigned int n_innersliding_nodes;
    /// Local indices of innersliding volume nodes.
    dealii::IndexSet innersliding_vol_range;
    /// Converts innersliding volume index to global index of volume nodes.
    dealii::LinearAlgebra::distributed::Vector<int> innersliding_vol_index_to_vol_index;
};

} // namespace PHiLiP
#endif
