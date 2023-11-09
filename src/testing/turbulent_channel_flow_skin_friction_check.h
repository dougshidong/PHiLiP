#ifndef __TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK__
#define __TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Turbulent Channel Flow Skin Friction Check
template <int dim, int nstate>
class TurbulentChannelFlowSkinFrictionCheck: public TestsBase
{
public:
    /// Constructor
    TurbulentChannelFlowSkinFrictionCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~TurbulentChannelFlowSkinFrictionCheck() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Half channel height
    const double half_channel_height;

    /// Run test
    int run_test () const override;
private:
    double get_x_velocity(const double y) const;
    double get_x_velocity_gradient(const double y) const;
    double compute_wall_shear_stress() const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
