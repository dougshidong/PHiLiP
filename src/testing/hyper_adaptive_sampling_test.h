#ifndef __HYPER_ADAPTIVE_SAMPLING_TEST_H__
#define __HYPER_ADAPTIVE_SAMPLING_TEST_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Hyper-reduction test, verifies accuracy of the solution with ECSW hyper-reduction of the residual and Jacobian
template <int dim, int nstate>
class HyperAdaptiveSamplingTest: public TestsBase
{
public:
    /// Constructor.
    HyperAdaptiveSamplingTest(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinitParams(const int max_iter) const;
    
    /// Run Hyper-reduction tes
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
