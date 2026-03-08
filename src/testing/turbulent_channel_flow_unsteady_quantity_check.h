#ifndef __TURBULENT_CHANNEL_FLOW_UNSTEADY_QUANTITY_CHECK__
#define __TURBULENT_CHANNEL_FLOW_UNSTEADY_QUANTITY_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Turbulent Channel Flow Unsteady Quantity Check
template <int dim, int nstate>
class TurbulentChannelFlowUnsteadyQuantityCheck: public TestsBase
{
public:
    /// Constructor
    TurbulentChannelFlowUnsteadyQuantityCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected average wall shear stress at final time
    const double average_wall_shear_stress_expected;

    /// Expected skin friction coefficient at final time
    const double skin_friction_coefficient_expected;

    /// Flag for using wall model
    const bool using_wall_model;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
