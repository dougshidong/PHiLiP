#ifndef __DIPOLE_WALL_COLLISION_UNSTEADY_QUANTITY_CHECK__
#define __DIPOLE_WALL_COLLISION_UNSTEADY_QUANTITY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Dipole Wall Collision Unsteady Quantity Check
template <int dim, int nstate>
class DipoleWallCollisionUnsteadyQuantityCheck: public TestsBase
{
public:
    /// Constructor
    DipoleWallCollisionUnsteadyQuantityCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Expected enstrophy at final time
    const double enstrophy_expected;

    /// Expected palinstrophy at final time
    const double palinstrophy_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
