#ifndef __DESIGN_PARAMETERIZATION_BASE_H__
#define __DESIGN_PARAMETERIZATION_BASE_H__

#include "mesh/high_order_grid.h"

namespace PHiLiP {
    
using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; 
using MatrixType = dealii::TrilinosWrappers::SparseMatrix;

/// Abstract class for design parameterization. Objective function and the constraints take this class's pointer as an input to parameterize design variables w.r.t. volume nodes.
template<int dim>
class DesignParameterizationBase {

public:
    /// Constructor
    DesignParameterizationBase(std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid);
    
    /// Destructor
    virtual ~DesignParameterizationBase();
    
    /// Initialize design variables and set locally owned and ghost indices.     
    virtual void initialize_design_variables(VectorType &design_var) = 0; 
    
    /// Checks if the design variable has changed and updates volume nodes based on the parameterization. 
    virtual bool update_mesh_from_design_variables( 
        const MatrixType &dXv_dXp,
        const VectorType &design_var) = 0;

    /// Computes derivative of volume nodes w.r.t. design parameters.
    virtual void compute_dXv_dXp(MatrixType &dXv_dXp) const = 0;

    /// Outputs design variables. Doesn't output anything if not overridden.
    virtual void output_design_variables(const unsigned int /*iteration_no*/) const;
    
    /// Returns the number of design variables. To be implemented by derived classes.
    virtual unsigned int get_number_of_design_variables() const = 0;

    /// Checks if the design variable has changed.
    bool has_design_variable_been_updated(const VectorType &current_design_var, const VectorType &updated_design_var) const;

    /// Pointer to high order grid
    std::shared_ptr<HighOrderGrid<dim,double>> high_order_grid;
    
    /// Alias for MPI_COMM_WORLD
    MPI_Comm mpi_communicator;

    /// std::cout only on processor #0.
    dealii::ConditionalOStream pcout;

    /// Processor# of current processor.
    int mpi_rank;

    /// Total no. of processors
    int n_mpi;
};

} // PHiLiP namespace

#endif
