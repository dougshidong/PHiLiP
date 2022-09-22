#ifndef __DESIGN_PARAMETERIZATION_FFD_H__
#define __DESIGN_PARAMETERIZATION_FFD_H__

#include "design_parameterization_base.hpp"

namespace PHiLiP {

/// FFD design parameterization. Holds an object of FreeFormDeformation and uses it to update the mesh when the control variables are updated.
template<int dim>
class DesignParameterizationFreeFormDeformation : public DesignParameterizationBase<dim> {
public:
    /// Constructor
    DesignParameterizationFreeFormDeformation(
        const FreeFormDeformation<dim> &_ffd,
        std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim);
    
    /// Destructor
    ~DesignParameterizationFreeFormDeformation(){} 
    
    /// Initializes FFD design variables and set locally owned and ghost indices. Overrides the virtual function in base class.
    void initialize_design_variables(VectorType &ffd_des_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. FFD design parameters. Overrides the virtual function in base class.
    void compute_dXv_dXp(const HighOrderGrid<dim,double> &high_order_grid, MatrixType &dXv_dXp) override;
    
    /// Checks if the design variables have changed and updates volume nodes based on the parameterization. 
    bool update_mesh_from_design_variables(
        HighOrderGrid<dim,double> &high_order_grid, 
        const MatrixType &dXv_dXp,
        const VectorType &ffd_des_var) override;

    /// Outputs design variables of FFD.
    void output_design_variables(const unsigned int iteration_no) override;
    
    /// Returns the number of FFD design variables.
    unsigned int get_number_of_design_variables();

private:
    /// Free-form deformation used to parametrize the geometry.
    FreeFormDeformation<dim> ffd;

    /// List of FFD design variables and axes.
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;

    /// Initial design variable. Value is set in initialize_design_variables().
    VectorType initial_ffd_des_var;
};

} // namespace PHiLiP
#endif
