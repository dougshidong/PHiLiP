#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
ReducedOrderModelParam::ReducedOrderModelParam () {}

void ReducedOrderModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        prm.declare_entry("adaptation_tolerance", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for POD adaptation");
        prm.declare_entry("path_to_search", ".",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Path to search for saved snapshots or POD basis.");
        prm.declare_entry("reduced_residual_tolerance", "1E-13",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for nonlinear reduced residual");
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        reduced_residual_tolerance = prm.get_double("reduced_residual_tolerance");
        path_to_search = prm.get("path_to_search");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace