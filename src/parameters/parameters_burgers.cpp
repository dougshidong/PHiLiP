#include "parameters_burgers.h"

namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
BurgersParam::BurgersParam () {}

void BurgersParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("burgers");
    {
        prm.declare_entry("rewienski_a", "2.2360679775", //sqrt(5)
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, dealii::Patterns::Double::max_double_value),
                          "Burgers Rewienski parameter a");
        prm.declare_entry("rewienski_b", "0.02",
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, dealii::Patterns::Double::max_double_value),
                          "Burgers Rewienski parameter b");
        prm.declare_entry("rewienski_manufactured_solution", "false",
                          dealii::Patterns::Bool(),
                          "Adds the manufactured solution source term to the PDE source term."
                          "Set as true for running a manufactured solution.");
        prm.declare_entry("diffusion_coefficient", "0.115572735", //Default value equal to 0.1*atan(1)*4.0/exp(1) (no fractions or square/cubic roots) chosen to prevent a test passing because of a cancelling ratio
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, dealii::Patterns::Double::max_double_value),
                          "Viscous Burgers diffusion coefficient");
    }
    prm.leave_subsection();
}

void BurgersParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("burgers");
    {
        rewienski_a = prm.get_double("rewienski_a");
        rewienski_b = prm.get_double("rewienski_b");
        rewienski_manufactured_solution = prm.get_bool("rewienski_manufactured_solution");
        diffusion_coefficient = prm.get_double("diffusion_coefficient");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace

