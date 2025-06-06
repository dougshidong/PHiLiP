#include "parameters/parameters_euler.h"

namespace PHiLiP {
namespace Parameters {

void EulerParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("euler");
    {
        prm.declare_entry("reference_length", "1.0",
                          dealii::Patterns::Double(),
                          "Reference length for non-dimensionalization.");
        prm.declare_entry("mach_infinity", "0.5",
                          dealii::Patterns::Double(1e-15, 2500),
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

        using convert_tensor = dealii::Patterns::Tools::Convert<dealii::Tensor<1, 5, double>>;
        prm.declare_entry("custom_boundary_for_each_state", "0,0,0,0,0", 
                          *convert_tensor::to_pattern(), 
                          "Custom boundary values for each primitive state. Used for cases that involve post shock boundaries or do not work with non-dimensionalization (ie. Shu Osher)");
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
        const double pi = atan(1.0) * 4.0;
        angle_of_attack = prm.get_double("angle_of_attack") * pi/180.0;
        side_slip_angle = prm.get_double("side_slip_angle") * pi/180.0;

        using convert_tensor = dealii::Patterns::Tools::Convert<dealii::Tensor<1, 5, double>>;
        custom_boundary_for_each_state = convert_tensor::to_value(prm.get("custom_boundary_for_each_state"));
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
