#include "parameters/parameters_mesh_adaptation.h"

namespace PHiLiP {
namespace Parameters {

MeshAdaptationParam::MeshAdaptationParam() {}

void MeshAdaptationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        prm.declare_entry("total_refinement_steps","0",
                          dealii::Patterns::Integer(),
                          "Maximum adaptation steps for a problem.");
        
        prm.declare_entry("critical_residual_val","1.0e-9",
                          dealii::Patterns::Double(0.0,1.0e5),
                          "Critical residual below which adaptation begins.");
        
        prm.declare_entry("use_goal_oriented_mesh_adaptation","false",
                          dealii::Patterns::Bool(),
                          "Flag to use goal oriented mesh adaptation. False by default.");

        prm.declare_entry("refinement_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be refined.");

        prm.declare_entry("coarsening_fraction","0.0",
                          dealii::Patterns::Double(0.0,1.0),
                          "Fraction of cells to be coarsened.");
    }
    prm.leave_subsection();

}

void MeshAdaptationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        total_refinement_steps = prm.get_integer("total_refinement_steps");
        critical_residual_val = prm.get_double("critical_residual_val");
        use_goal_oriented_mesh_adaptation = prm.get_bool("use_goal_oriented_mesh_adaptation");
        refinement_fraction = prm.get_double("refinement_fraction");
        coarsening_fraction = prm.get_double("coarsening_fraction");
    }
    prm.leave_subsection();
}

} // namespace Parameters
} // namespace PHiLiP

