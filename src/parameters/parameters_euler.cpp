#include "parameters/parameters_euler.h"

namespace PHiLiP {
namespace Parameters {

// Euler inputs
EulerParam::EulerParam () {}

void EulerParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("euler");
    {
        prm.declare_entry("reference_length", "1.0",
                          dealii::Patterns::Double(),
                          "Reference length for non-dimensionalization.");
        prm.declare_entry("mach_infinity", "1.0",
                          dealii::Patterns::Double(1e-15, 10),
                          "Farfield Mach number");
        prm.declare_entry("gamma_gas", "1.4",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Gamma gas constant");
        prm.declare_entry("angle_of_attack", "0.0",
                          dealii::Patterns::Double(-180, 180),
                          "Angle of attack in degrees. Required for 2D");
        prm.declare_entry("side_slip_angle", "0.0",
                          dealii::Patterns::Double(-180, 180),
                          "Side slip angle in degrees. Required for 3D");
    }
    prm.leave_subsection();
}

void EulerParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("euler");
    {
        ref_length      = prm.get_double("reference_length");
        mach_inf        = prm.get_double("mach_infinity");
        gamma_gas       = prm.get_double("gamma_gas");
        angle_of_attack = prm.get_double("angle_of_attack");
        side_slip_angle = prm.get_double("side_slip_angle");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
