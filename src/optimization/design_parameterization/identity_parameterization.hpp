#ifndef __DESIGN_PARAMETERIZATION_IDENTITY_H__
#define __DESIGN_PARAMETERIZATION_IDENTITY_H__

#include "base_parameterization.hpp"

namespace PHiLiP {

/// Identity design parameterization. Control variables are all volume nodes.
template<int dim>
class DesignParameterizationIdentity : public DesignParameterizationBase<dim> {
public:
    /// Constructor
    DesignParameterizationIdentity(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid); 
    
    /// Destructor
    ~DesignParameterizationIdentity() override {} 
    
    /// Initializes design variables with volume nodes and set locally owned and ghost indices. Overrides the virtual function in base class.
    void initialize_design_variables(VectorType &design_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. design parameters. Overrides the virtual function in base class.
    /** As the volume nodes are the design parameters, dXv_dXp is identity.
     */
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    /// Checks if the design variables have changed and updates volume nodes. 
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    /// Returns the number of design variables (i.e. total no. of volume nodes on all processors).
    unsigned int get_number_of_design_variables() const override;

private:
    VectorType current_volume_nodes;
};

} // namespace PHiLiP
#endif
