#include "hyper_adaptive_sampling_new_error.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/assemble_ECSW_residual.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "reduced_order/pod_adaptive_sampling.h"
#include "reduced_order/hyper_reduced_sampling_error_updated.h"
#include <iostream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
HyperAdaptiveSamplingNewError<dim, nstate>::HyperAdaptiveSamplingNewError(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int HyperAdaptiveSamplingNewError<dim, nstate>::run_test() const
{
    pcout << "Starting hyperreduced adaptive sampling procedure..." << std::endl;

    HyperreducedSamplingErrorUpdated<dim, nstate> sampling_imp(this->all_parameters, parameter_handler);
    int exit = sampling_imp.run_sampling();
    
    if (exit == 0){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

#if PHILIP_DIM==1
        template class HyperAdaptiveSamplingNewError<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperAdaptiveSamplingNewError<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
