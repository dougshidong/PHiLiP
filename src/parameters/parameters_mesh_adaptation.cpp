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
        
        prm.declare_entry("use_goal_oriented_mesh_adaptation","false",
                          dealii::Patterns::Bool(),
                          "Flag to use goal oriented mesh adaptation. False by default.");

        prm.declare_entry("h_refine_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be h-refined.");

        prm.declare_entry("h_coarsen_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be h-coarsened.");
        
        prm.declare_entry("p_refine_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be p-refined.");

        prm.declare_entry("p_coarsen_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be p-coarsened.");
    }
    prm.leave_subsection();

}

void MeshAdaptationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        total_mesh_adaptation_cycles = prm.get_integer("total_mesh_adaptation_cycles");
        use_goal_oriented_mesh_adaptation = prm.get_bool("use_goal_oriented_mesh_adaptation");
        h_refine_fraction = prm.get_double("h_refine_fraction");
        h_coarsen_fraction = prm.get_double("h_coarsen_fraction");
        p_refine_fraction = prm.get_double("p_refine_fraction");
        p_coarsen_fraction = prm.get_double("p_coarsen_fraction");
    }
    prm.leave_subsection();
}

} // namespace Parameters
} // namespace PHiLiP

