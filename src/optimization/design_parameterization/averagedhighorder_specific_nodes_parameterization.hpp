#ifndef __SPECIFIC_NODES_PARAMETERIZATION_H__
#define __SPECIFIC_NODES_PARAMETERIZATION_H__

#include "base_parameterization.hpp"

namespace PHiLiP {

template<int dim>
class SpecificNodesParameterization : public BaseParameterization<dim> {
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    SpecificNodesParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid); 
    
    /// Destructor
    ~SpecificNodesParameterization() {};
    
    void initialize_design_variables(VectorType &design_var) override;
    
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    unsigned int get_number_of_design_variables() const override;
    
    int is_design_variable_valid(const MatrixType &dXv_dXp, const VectorType &design_var) const override;
    
    void compute_control_index_to_vol_index();

    bool check_if_node_belongs_to_the_line(const double x) const;

private:
    
    VectorType current_design_var;
    
    unsigned int n_control_nodes;
    
    dealii::IndexSet control_index_range;
    
    dealii::IndexSet control_ghost_range;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_left_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_right_vol_index;
};

} // namespace PHiLiP
#endif
