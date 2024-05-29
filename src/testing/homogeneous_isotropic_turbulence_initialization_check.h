#ifndef __homogeneous_isotropic_turbulence_initialization_check__
#define __homogeneous_isotropic_turbulence_initialization_check__

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex Restart Check
template <int dim, int nstate>
class HomogeneousIsotropicTurbulenceInitializationCheck: public TestsBase
{
public:
    /// Constructor
    HomogeneousIsotropicTurbulenceInitializationCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
