#ifndef __DESIGN_PARAMETERIZATION_INNERVOL_H__
#define __DESIGN_PARAMETERIZATION_INNERVOL_H__

#include "base_parameterization.hpp"

namespace PHiLiP {

/// Design parameterization w.r.t. inner volume nodes (i.e. volume nodes excluding those on the boundary).
template<int dim>
class InnerVolParameterization : public BaseParameterization<dim> {
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    InnerVolParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid); 
    
    /// Destructor
    ~InnerVolParameterization() {};
    
    /// Initializes design variables with inner volume nodes and set locally owned and ghost indices. Overrides the virtual function in base class.
    void initialize_design_variables(VectorType &design_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. inner volume nodes. Overrides the virtual function in base class.
    /** dXv_dXp is a rectangular matrix of dimension n_vol_nodes x n_inner_nodes.
     */
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    /// Checks if the design variables have changed and updates inner volume nodes of high order grid. 
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    /// Returns the number of design variables (i.e. no. of inner volume nodes).
    unsigned int get_number_of_design_variables() const override;
    
    /// Computes inner_vol_range IndexSet on each processor, along with n_inner_nodes and inner_vol_index_to_vol_index. 
    /** Called by the constructor of the class. These variables are expected to be constant throughtout the optimization for now (as no new nodes are being added).
     */
    void compute_inner_vol_index_to_vol_index();

private:
    /// Current design variable. Stored to prevent recomputing if it is unchanged. 
    VectorType current_design_var;
    /// No. of inner volume nodes.
    unsigned int n_inner_nodes;
    /// Local indices of inner volume nodes.
    dealii::IndexSet inner_vol_range;
    /// Converts inner volume index to global index of volume nodes.
    dealii::LinearAlgebra::distributed::Vector<int> inner_vol_index_to_vol_index;
};

} // namespace PHiLiP
#endif
