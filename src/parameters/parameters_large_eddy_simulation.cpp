#include "parameters/parameters_large_eddy_simulation.h"

namespace PHiLiP {
namespace Parameters {
    
// LargeEddySimulation inputs
LargeEddySimulationParam::LargeEddySimulationParam () {}

void LargeEddySimulationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("large_eddy_simulation");
    {
        prm.declare_entry("SGS_model_type", "smagorinsky",
                           dealii::Patterns::Selection(
                          " smagorinsky | "
                          " wall_adaptive_local_eddy_viscosity |"
                          " dynamic_smagorinsky"),
                          "Enum of sub-grid scale models."
                          "Choices are "
                          " <smagorinsky | "
                          "  wall_adaptive_local_eddy_viscosity | "
                          "  dynamic_smagorinsky>.");

        prm.declare_entry("turbulent_prandtl_number", "0.6",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Turbulent Prandlt number");

        prm.declare_entry("smagorinsky_model_constant", "0.1",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Smagorinsky model constant");
    }
    prm.leave_subsection();
}

void LargeEddySimulationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("large_eddy_simulation");
    {
        const std::string SGS_model_type_string = prm.get("SGSmodel_type");
        if(SGS_model_type_string == "smagorinsky") {
            SGS_model_type = smagorinsky;
        } else if(SGS_model_type_string == "wall_adaptive_local_eddy_viscosity") {
            SGS_model_type = wall_adaptive_local_eddy_viscosity;
        } else if(SGS_model_type_string == "dynamic_smagorinsky") {
            SGS_model_type = dynamic_smagorinsky;
        }

        turbulent_prandtl_number = prm.get_double("turbulent_prandtl_number");
        smagorinsky_model_constant = prm.get_double("smagorinsky_model_constant");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
