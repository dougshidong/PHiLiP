#ifndef __DESIGN_PARAMETERIZATION_BASE_H__
#define __DESIGN_PARAMETERIZATION_BASE_H__

#include "mesh/free_form_deformation.h"

namespace PHiLiP {
    
using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; 
using MatrixType = dealii::TrilinosWrappers::SparseMatrix;

/// Abstract class for design parameterization. Objective function and the constraints take this class's pointer as an input to parameterize design variables w.r.t. volume nodes.
template<int dim>
class DesignParameterizationBase {

public:
    /// Constructor
    DesignParameterizationBase(){}
    /// Destructor
    virtual ~DesignParameterizationBase(){}
    
    /// Initialize design variables and set locally owned and ghost indices.     
    virtual void initialize_design_variables(VectorType &design_var) = 0; 
    
    /// Checks if the design variable has changed and updates volume nodes based on the parameterization. 
    virtual bool update_mesh_from_design_variables(
        HighOrderGrid<dim,double> &high_order_grid, 
        const MatrixType &dXv_dXp,
        const VectorType &design_var) = 0;

    /// Computes derivative of volume nodes w.r.t. design parameters.
    virtual void compute_dXv_dXp(const HighOrderGrid<dim,double> &high_order_grid, MatrixType &dXv_dXp) = 0;

    /// Outputs design variables. Doesn't output anything if not overridden.
    virtual void output_design_variables(const unsigned int /*iteration_no*/) {}
    
    /// Returns the number of design variables. To be implemented by derived classes.
    virtual unsigned int get_number_of_design_variables() = 0;
};

} // PHiLiP namespace

#endif
