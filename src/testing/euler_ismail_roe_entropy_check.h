#ifndef __EULER_ISLAIL_ROE_ENTROPY_CHECK__
#define __EULER_ISLAIL_ROE_ENTROPY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Euler Ismail-Roe Entropy Check
template <int dim, int nstate>
class EulerIsmailRoeEntropyCheck: public TestsBase
{
public:
    /// Constructor
    EulerIsmailRoeEntropyCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~EulerIsmailRoeEntropyCheck() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
