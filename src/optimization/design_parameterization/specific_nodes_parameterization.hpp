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

    bool check_if_node_belongs_to_the_region(const double x, const double y) const;

    double get_slope_y(const unsigned int ivol, const bool on_boundary) const;
    
    void store_prespecified_control_nodes();
    
    void write_control_nodes_to_file(const std::vector<std::pair<double,double>> &final_control_nodes_list) const;
    std::vector<std::pair<double,double>> get_final_control_nodes_list() const;
    void output_control_nodes() const override;
    void output_control_nodes_with_interpolated_high_order_nodes() const override;
    void output_control_nodes_refined() const override;

private:
    
    VectorType current_design_var;
    
    unsigned int n_control_nodes;
    
    dealii::IndexSet control_index_range;
    
    dealii::IndexSet control_ghost_range;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_left_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> control_index_to_right_vol_index;
    
    dealii::LinearAlgebra::distributed::Vector<int> is_on_boundary;
    
    std::vector<double> x_control_nodes;
    std::vector<double> y_control_nodes;
};

} // namespace PHiLiP
#endif
