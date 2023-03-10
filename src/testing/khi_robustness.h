#ifndef __KHI_ROBUSTNESS__
#define __KHI_ROBUSTNESS__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// KHI Robustness test
/** Runs the Kelvin-Helmholtz Instability (KHI) test case
 * until a crash is detected, then restart a new 
 * simulation with a different Atwood number.
 */
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

    /// Reinit parameters based on a specified Atwood number
    /** Atwood number quantifies the density difference
     * A = \frac{\rho_2-\rho1}{\rho_1+\rho_2}
     */
    Parameters::AllParameters reinit_params(double atwood_number) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
