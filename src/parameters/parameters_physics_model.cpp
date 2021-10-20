#include "parameters/parameters_physics_model.h"

namespace PHiLiP {
namespace Parameters {
    
// Models inputs
PhysicsModelParam::PhysicsModelParam () {}

void PhysicsModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("physics_models");
    {
        prm.declare_entry("baseline_physics_type", "navier_stokes",
                          dealii::Patterns::Selection(
                            " euler | "
                            " navier_stokes"),
                            "Enum of models."
                            "Choices are "
                            " <euler | "
                            "  navier_stokes>.");

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
    prm.leave_subsection();
}

void PhysicsModelParam::parse_parameters (dealii::ParameterHandler &prm, const std::string physics_model_string)
{
    prm.enter_subsection("physics_model");
    {
        const std::string baseline_physics_string = prm.get("baseline_physics_type");
        if(baseline_physics_string == "euler")         baseline_physics_type = BaselinePhysicsEnum::euler;
        if(baseline_physics_string == "navier_stokes") baseline_physics_type = BaselinePhysicsEnum::navier_stokes;

        if(physics_model_string == "large_eddy_simulation")
        {
            prm.enter_subsection("large_eddy_simulation");
            {
                const std::string SGS_model_type_string = prm.get("SGSmodel_type");
                if(SGS_model_type_string == "smagorinsky")                        SGS_model_type = smagorinsky;
                if(SGS_model_type_string == "wall_adaptive_local_eddy_viscosity") SGS_model_type = wall_adaptive_local_eddy_viscosity;
                if(SGS_model_type_string == "dynamic_smagorinsky")                SGS_model_type = dynamic_smagorinsky;

                turbulent_prandtl_number = prm.get_double("turbulent_prandtl_number");
                smagorinsky_model_constant = prm.get_double("smagorinsky_model_constant");
            }
            prm.leave_subsection();
        }
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
