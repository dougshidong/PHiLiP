#ifndef __HALTON_SAMPLING_RUN_H__
#define __HALTON_SAMPLING_RUN_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Runs adaptive sampling procedure 
template <int dim, int nstate>
class HaltonSamplingRun: public TestsBase
{
public:
    /// Constructor.
    HaltonSamplingRun(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Run adaptive sampling procedure 
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
