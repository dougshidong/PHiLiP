#include "hyper_adaptive_sampling_test.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "hyper_reduction/assemble_problem_ECSW.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "pod_adaptive_sampling.h"
#include "hyper_reduction/hyper_reduced_adaptive_sampling.h"
#include <iostream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
HyperAdaptiveSamplingTest<dim, nstate>::HyperAdaptiveSamplingTest(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters HyperAdaptiveSamplingTest<dim, nstate>::reinitParams(const int max_iter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.ode_solver_param.nonlinear_max_iterations = max_iter;
    return parameters;
}

template <int dim, int nstate>
int HyperAdaptiveSamplingTest<dim, nstate>::run_test() const
{
    pcout << "Starting hyper-reduction adaptive sampling test..." << std::endl;

    HyperReduction::HyperreducedAdaptiveSampling<dim, nstate> sampling_imp(this->all_parameters, parameter_handler);
    int exit = sampling_imp.run();
    
    if (exit == 0){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

#if PHILIP_DIM==1
        template class HyperAdaptiveSamplingTest<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperAdaptiveSamplingTest<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
