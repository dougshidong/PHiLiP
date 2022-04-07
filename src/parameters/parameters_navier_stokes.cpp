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
        prm.declare_entry("isothermal_wall_temperature", "1.0",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Isothermal wall temperature.");
        prm.declare_entry("wall_temperature_boundary_condition_type", "adiabatic",
                          dealii::Patterns::Selection("adiabatic|isothermal"),
                          "Type of wall temperature boundary conditions to be imposed. "
                          "Choices are <adiabatic|isothermal>.");
    }
    prm.leave_subsection();
}

void NavierStokesParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("navier_stokes");
    {
        prandtl_number              = prm.get_double("prandtl_number");
        reynolds_number_inf         = prm.get_double("reynolds_number_inf");
        isothermal_wall_temperature = prm.get_double("isothermal_wall_temperature");

        const std::string wall_temperature_boundary_condition_type_string = prm.get("wall_temperature_boundary_condition_type");
        if (wall_temperature_boundary_condition_type_string == "adiabatic")  wall_temperature_boundary_condition_type = WallTemperatureBoundaryConditionEnum::adiabatic;
        if (wall_temperature_boundary_condition_type_string == "isothermal") wall_temperature_boundary_condition_type = WallTemperatureBoundaryConditionEnum::isothermal;
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
