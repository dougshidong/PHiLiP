#ifndef __TAYLOR_GREEN_VORTEX_RESTART_CHECK__
#define __TAYLOR_GREEN_VORTEX_RESTART_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex Restart Check
template <int dim, int nstate>
class TaylorGreenVortexRestartCheck: public TestsBase
{
public:
    /// Constructor
    TaylorGreenVortexRestartCheck(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~TaylorGreenVortexRestartCheck() {};
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
