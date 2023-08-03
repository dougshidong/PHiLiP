#include "parameters/parameters_potential_source.h"

namespace PHiLiP {
namespace Parameters {

// Potential source geometry inputs
PotentialSourceParam::PotentialSourceParam () {}

void PotentialSourceParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("potential_source");
    {
        prm.declare_entry("potential_source_geometry", "none",
                      dealii::Patterns::Selection("trailing_edge_serrations|circular_test|none"),
                      "Choose geometry used for potential source term. "
                      "Choices are <trailing_edge_serrations|circular_test|none>.");

        prm.declare_entry("viscous_drag","true",
                      dealii::Patterns::Bool(),
                      "Set as true by default (i.e. apply viscous drag term). " 
                      "If true, includes the contribution of the viscous drag.");

        prm.enter_subsection("trailing edge serrations");
        {
            prm.declare_entry("trailing_edge_serration_frequency", "0.015",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Distance between consecutive trailing edge serrations.");
            prm.declare_entry("half_trailing_edge_serration_length", "0.015",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Half the length of trailing edge serrations.");
            prm.declare_entry("trailing_edge_serration_thickness", "0.001",
                      dealii::Patterns::Double(1e-15, 1000),
                      "Thickness of trailing edge serrations.");
            prm.declare_entry("trailing_edge_serration_effective_length_factor", "0.5",
                              dealii::Patterns::Double(1e-15, 1),
                              "Effective length factor for trailing edge serrations.");
            prm.declare_entry("angle_of_serrations", "0.0",
                              dealii::Patterns::Double(-180, 180),
                              "Angle of trailing edge flap in degrees (gamma).");
        }
        prm.leave_subsection();

        prm.enter_subsection("circular test");
        {
            prm.declare_entry("circle_radius", "1.0",
                              dealii::Patterns::Double(1e-15, 10000000),
                              "Radius of 2D circle test case.");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

}

void PotentialSourceParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("potential_source");
    {
        use_viscous_drag = prm.get_bool("viscous_drag");

        const std::string geometry_string = prm.get("potential_source_geometry");
        if (geometry_string == "trailing_edge_serrations") potential_source_geometry = trailing_edge_serrations;
        if (geometry_string == "circular_test") potential_source_geometry = circular_test;
        if (geometry_string == "none") potential_source_geometry = none;

        if (geometry_string == "trailing_edge_serrations")
        {
            prm.enter_subsection("trailing edge serrations");
            {
                TES_frequency   = prm.get_double("trailing_edge_serration_frequency");
                TES_h           = prm.get_double("half_trailing_edge_serration_length");
                TES_thickness   = prm.get_double("trailing_edge_serration_thickness");
                TES_effective_length_factor = prm.get_double("trailing_edge_serration_effective_length_factor");
                const double pi = atan(1.0) * 4.0;
                TES_flap_angle       = prm.get_double("angle_of_serrations") * pi/180.0;
            }
            prm.leave_subsection();
        }

        if (geometry_string == "circular_test")
        {
            prm.enter_subsection("circular test");
            {
                circle_radius   = prm.get_double("circle_radius");
            }
            prm.leave_subsection();
        }
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
