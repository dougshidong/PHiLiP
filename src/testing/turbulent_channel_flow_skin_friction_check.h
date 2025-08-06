#ifndef __TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK__
#define __TURBULENT_CHANNEL_FLOW_SKIN_FRICTION_CHECK__

#include "tests.h"
#include "parameters/parameters_flow_solver.h"

namespace PHiLiP {
namespace Tests {

/// Turbulent Channel Flow Skin Friction Check
template <int dim, int nstate>
class TurbulentChannelFlowSkinFrictionCheck: public TestsBase
{
private:
    /// Enumeration of all turbulent channel flow initial condition sub-types defined in the Parameters class
    using XVelocityInitialConditionEnum = Parameters::FlowSolverParam::XVelocityInitialConditionType;
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

    // Turbulent channel x-velocity initial condition type
    const XVelocityInitialConditionEnum xvelocity_initial_condition_type;

    const double y_top_wall; // y-value for top wall
    const double y_bottom_wall; // y-value for bottom wall
    const double normal_vector_top_wall; // normal vector for top wall
    const double normal_vector_bottom_wall; // normal vector for bottom wall

    /// Run test
    int run_test () const override;
private:
    double get_x_velocity(const double y) const;
    double get_x_velocity_gradient(const double y) const;
    double get_wall_shear_stress() const;
    double get_bulk_velocity() const;
    double get_skin_friction_coefficient() const;
    
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
