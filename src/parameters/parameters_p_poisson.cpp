#include "parameters/parameters_p_poisson.h"

namespace PHiLiP {
namespace Parameters {

// p-Poisson inputs
PPoissonParam::PPoissonParam () {}

void PPoissonParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("p_poisson");
    {
        prm.declare_entry("factor_p", "4.0",
                          dealii::Patterns::Double(0.0, 1000.0),
                          "Factor p in p-Poisson model. Default value is 4. ");
        prm.declare_entry("stable_factor", "0.01",
                          dealii::Patterns::Double(0.0, 1000.0),
                          "Stable factor in p-Poisson model. Default value is 0.01. ");
    }
    prm.leave_subsection();
}

void PPoissonParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("p_poisson");
    {
        factor_p      = prm.get_double("factor_p");
        stable_factor = prm.get_double("stable_factor");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace