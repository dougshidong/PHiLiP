#include "parameters/parameters_mesh_adaptation.h"

namespace PHiLiP {
namespace Parameters {

MeshAdaptationParam::MeshAdaptationParam() {}

void MeshAdaptationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        prm.declare_entry("total_mesh_adaptation_cycles","0",
                           dealii::Patterns::Integer(),
                          "Maximum adaptation steps for a problem.");

        prm.declare_entry("mesh_adaptation_type", "h_adaptation",
                          dealii::Patterns::Selection(
                          " h_adaptation | "
                          " p_adaptation | "
                          " hp_adaptation "
                          ),
                          "Mesh adaptation type that we want to use. "
                          "Choices are "
                          " <h_adaptation | "
                          "  p_adaptation | "
                          "  hp_adaptation>.");
        
        prm.declare_entry("use_goal_oriented_mesh_adaptation","false",
                          dealii::Patterns::Bool(),
                          "Flag to use goal oriented mesh adaptation. False by default.");

        prm.declare_entry("h_refine_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be h-refined.");

        prm.declare_entry("h_coarsen_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be h-coarsened.");

        prm.declare_entry("hp_smoothness_tolerance","1.0e-6",
                          dealii::Patterns::Double(0.0,1.0e5),
                          "Tolerance to decide between h- or p-refinement.");
    }
    prm.leave_subsection();

}

void MeshAdaptationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        total_mesh_adaptation_cycles = prm.get_integer("total_mesh_adaptation_cycles");
        const std::string mesh_adaptation_string = prm.get("mesh_adaptation_type");
        if(mesh_adaptation_string == "h_adaptation")       {mesh_adaptation_type = MeshAdaptationType::h_adaptation;}
        else if(mesh_adaptation_string == "p_adaptation")  {mesh_adaptation_type = MeshAdaptationType::p_adaptation;}
        else if(mesh_adaptation_string == "hp_adaptation") {mesh_adaptation_type = MeshAdaptationType::hp_adaptation;}
        
        use_goal_oriented_mesh_adaptation = prm.get_bool("use_goal_oriented_mesh_adaptation");
        h_refine_fraction = prm.get_double("h_refine_fraction");
        h_coarsen_fraction = prm.get_double("h_coarsen_fraction");
        hp_smoothness_tolerance = prm.get_double("hp_smoothness_tolerance");
    }
    prm.leave_subsection();
}

} // namespace Parameters
} // namespace PHiLiP

