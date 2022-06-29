#ifndef __TAYLOR_GREEN_VORTEX_ENERGY_CHECK__
#define __TAYLOR_GREEN_VORTEX_ENERGY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex Energy Check
template <int dim, int nstate>
class TaylorGreenVortexEnergyCheck: public TestsBase
{
public:
    /// Constructor
    TaylorGreenVortexEnergyCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~TaylorGreenVortexEnergyCheck() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Expected theoretical dissipation rate at final time
    const double theoretical_dissipation_rate_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
