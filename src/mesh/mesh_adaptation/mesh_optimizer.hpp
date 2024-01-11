#ifndef __MESH_OPTIMIZER_H__
#define __MESH_OPTIMIZER_H__

#include "dg/dg.h"
#include "optimization/design_parameterization/base_parameterization.hpp"
#include "functional/functional.h"
#include <deal.II/optimization/rol/vector_adaptor.h>

namespace PHiLiP {

/// Class to run optimizer
template <int dim, int nstate>
class MeshOptimizer {

    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii distributed vector.
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;

public:
// Member functions
    /// Constructor.
    MeshOptimizer(std::shared_ptr<DGBase<dim,double>> dg_input,
                  const Parameters::AllParameters *const parameters_input, 
                  const bool _use_full_space_method);
    
    /// Destructor.
    ~MeshOptimizer(){}
    
    /// Runs full-space optimizer.
    void run_full_space_optimizer(const dealii::TrilinosWrappers::SparseMatrix &regularization_matrix);

    /// Gets parlist according to optimization_param.
    Teuchos::ParameterList get_parlist();
/*
    /// Runs reduced-space optimizer.
    void run_reduced_space_optimizer();

    /// Assings objective function and design parameterization.
    void initialize_objfunc_and_design_parameterization();

    void initialize_state_design_and_dual_variables();

    void initialize_output_stream();

    /// Checks derivatives.
    void check_derivatives();
*/
// Member variables
    /// Pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;

    /// Holds all parameters.
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    
    /// Flag to use full-space optimizer
    const bool use_full_space_method;
/*    
    /// Objective function
    std::shared_ptr<Functional<dim, nstate, double>> objective_function;

    /// Design parameterization
    std::shared_ptr<BaseParameterization<dim>> design_parameterization;

    /// State space steady-state solution u of the PDE.
    DealiiVector state_variables;

    /// Variables for which the optimization is performed.
    DealiiVector design_variables;

    /// Dual variables
    DealiiVector dual_variables;
    
    /// Outstream of Trilinos.
    Teuchos::RCP<std::ostream> rcp_outstream;
    
    ROL::nullstream null_stream; ///< To output nothing.
    
    std::filebuf filebuffer;

    std::ostream std_outstream;
*/    
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
