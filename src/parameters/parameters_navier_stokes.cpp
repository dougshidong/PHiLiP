#include "parameters/parameters_navier_stokes.h"

namespace PHiLiP {
namespace Parameters {
    
// NavierStokes inputs
NavierStokesParam::NavierStokesParam () {}

void NavierStokesParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("navier_stokes");
    {
        prm.declare_entry("prandtl_number", "0.72",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Prandlt number");
        prm.declare_entry("reynolds_number_inf", "10000000.0",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Farfield Reynolds number");
    }
    prm.leave_subsection();
}

void NavierStokesParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("navier_stokes");
    {
        prandtl_number      = prm.get_double("prandtl_number");
        reynolds_number_inf = prm.get_double("reynolds_number_inf");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
