#ifndef __MESH_OPTIMIZER_H__
#define __MESH_OPTIMIZER_H__

#include "dg/dg.h"
#include "optimization/design_parameterization/base_parameterization.hpp"
#include "functional/functional.h"

namespace PHiLiP {

template <int dim, int nstate>
class MeshOptimizer {

public:
// Member functions

    /// Constructor.
    MeshOptimizer(std::shared_ptr<DGBase<dim,double>> dg_input,
                  const Parameters::AllParameters *const parameters_input, 
                  const bool _use_full_space_method);
    
    /// Destructor.
    ~MeshOptimizer();
    
    /// Runs full-space optimizer.
    void run_full_space_optimizer();

    /// Runs reduced-space optimizer.
    void run_reduced_space_optimizer();

    /// Gets parlist according to optimization_param.
    Teuchos::ParameterList get_parlist();

    /// Assings objective function and design parameterization.
    void assign_functional_and_design_parameterization();

    /// Checks derivatives.
    void check_derivatives();

// Member variables
    /// Holds all parameters.
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    
    /// Pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;

    /// Flag to use full-space optimizer
    const bool use_full_space_method;

    /// Objective function
    std::shared_ptr< Functional<dim, nstate, double> > objective_function;

    /// Design parameterization
    std::shared_ptr<BaseParameterization<dim>> design_parameterization;

};

} // PHiLiP namespace

#endif
