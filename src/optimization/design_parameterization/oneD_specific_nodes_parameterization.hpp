#ifndef __ONED_SPECIFIC_NODES_PARAMETERIZATION_H__
#define __ONED_SPECIFIC_NODES_PARAMETERIZATION_H__

#include "base_parameterization.hpp"

namespace PHiLiP {

template<int dim>
class OneDSpecificNodesParameterization : public BaseParameterization<dim> {
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    OneDSpecificNodesParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid); 
    
    /// Destructor
    ~OneDSpecificNodesParameterization() {};
    
    void initialize_design_variables(VectorType &design_var) override;
    
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    unsigned int get_number_of_design_variables() const override;
    
    int is_design_variable_valid(const MatrixType &dXv_dXp, const VectorType &design_var) const override;
    
    void compute_control_index_to_vol_index();

    int check_if_node_belongs_to_the_region(const double x, const double y) const;
    
    void store_prespecified_control_nodes();

private:
    
    VectorType current_design_var;

    VectorType slope_vals;
    
    unsigned int n_control_nodes;
    
    dealii::IndexSet control_index_range;
    
    dealii::IndexSet control_ghost_range;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_left_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_right_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> is_on_boundary;
    
    std::vector<std::pair<double,double>> control_nodes_list;
    std::vector<double> control_slope;
};

} // namespace PHiLiP
#endif
