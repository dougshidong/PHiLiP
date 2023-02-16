#ifndef __EULER_ISMAIL_ROE_ENTROPY_CHECK__
#define __EULER_ISMAIL_ROE_ENTROPY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Euler Entropy Check for Split Forms
/** This test verifies behaviour for the split forms currently implemented.
 * Entropy should be conserved for IR, CH, Ra fluxes.
 * Entropy is not conserved by KG, but a tolerance has been set based on the 
 * expected behaviour of the test.
 */
template <int dim, int nstate>
class EulerSplitEntropyCheck: public TestsBase
{
public:
    /// Constructor
    EulerSplitEntropyCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~EulerSplitEntropyCheck() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
