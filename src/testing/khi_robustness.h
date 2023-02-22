#ifndef __KHI_ROBUSTNESS__
#define __KHI_ROBUSTNESS__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Euler Ismail-Roe Entropy Check
template <int dim, int nstate>
class KHIRobustness: public TestsBase
{
public:
    /// Constructor
    KHIRobustness(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~KHIRobustness() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:
    Parameters::AllParameters reinit_params(double atwood_number) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
