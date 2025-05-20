#ifndef __BUILD_NNLS_PROBLEM_H__
#define __BUILD_NNLS_PROBLEM_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include "linear_solver/helper_functions.h"

namespace PHiLiP {
namespace Tests {

/// Test assembling NNLS problem from Online POD
/// Note: An instance of AdaptiveSampling is built, but the sampling is not run to completion, only the initial snapshots are placed
/// Results compared to NNLS solution for MATLAB
template <int dim, int nstate>
class BuildNNLSProblem: public TestsBase
{
public:
    /// Constructor.
    BuildNNLSProblem(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Run Assemble Problem ECSW
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
