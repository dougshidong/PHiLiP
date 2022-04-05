#ifndef __TAYLOR_GREEN_VORTEX_ENERGY_CHECK__
#define __TAYLOR_GREEN_VORTEX_ENERGY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex
template <int dim, int nstate>
class TaylorGreenVortexEnergyCheck: public TestsBase
{
public:
    /// Constructor
    TaylorGreenVortexEnergyCheck(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~TaylorGreenVortexEnergyCheck() {};
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
