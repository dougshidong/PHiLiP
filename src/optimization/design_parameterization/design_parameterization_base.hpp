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
    DesignParameterizationBase(const unsigned int _n_design_variables);
    /// Destructor
    ~DesignParameterizationBase(){}
    
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
    
    /// Stores the number of design variables.
    const unsigned int n_design_variables;
};

template<int dim>
class DesignParameterizationFreeFormDeformation : public DesignParameterizationBase<dim> {
public:
    /// Constructor
    DesignParameterizationFreeFormDeformation(
        const FreeFormDeformation<dim> &_ffd,
        std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim);
    
    /// Destructor
    ~DesignParameterizationFreeFormDeformation(){}
    
    /// Initialize design variables and set locally owned and ghost indices. Overrides the virtual function in base class.
    void initialize_design_variables(VectorType &ffd_des_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. FFD design parameters. Overrides the virtual function in base class.
    void compute_dXv_dXp(const HighOrderGrid<dim,double> &high_order_grid, MatrixType &dXv_dXp) override;
    
    /// Checks if the design variable has changed and updates volume nodes based on the parameterization. 
    bool update_mesh_from_design_variables(
        HighOrderGrid<dim,double> &high_order_grid, 
        const MatrixType &dXv_dXp,
        const VectorType &ffd_des_var) override;

    /// Outputs design variables of FFD.
    void output_design_variables(const unsigned int iteration_no) override;

private:
    /// Free-form deformation used to parametrize the geometry.
    FreeFormDeformation<dim> ffd;

    /// List of FFD design variables and axes.
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;

    /// Initial design variable. Value is set in initialize_design_variables().
    VectorType initial_ffd_des_var;
};
} // PHiLiP namespace

#endif
