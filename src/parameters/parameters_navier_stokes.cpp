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
                          dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                          "Prandlt number. Default value is 0.72. "
                          "NOTE: Must be consitent with temperature_inf.");
        prm.declare_entry("reynolds_number_inf", "10000000.0",
                          dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                          "Farfield Reynolds number");
        prm.declare_entry("temperature_inf", "273.15",
                          dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                          "Farfield temperature in degree Kelvin [K]. Default value is 273.15K. "
                          "NOTE: Must be consistent with specified Prandtl number.");
        prm.declare_entry("nondimensionalized_isothermal_wall_temperature", "1.0",
                          dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                          "Nondimensionalized isothermal wall temperature.");
        prm.declare_entry("thermal_boundary_condition_type", "adiabatic",
                          dealii::Patterns::Selection("adiabatic|isothermal"),
                          "Type of thermal boundary conditions to be imposed. "
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
        temperature_inf             = prm.get_double("temperature_inf");
        nondimensionalized_isothermal_wall_temperature = prm.get_double("nondimensionalized_isothermal_wall_temperature");

        const std::string thermal_boundary_condition_type_string = prm.get("thermal_boundary_condition_type");
        if (thermal_boundary_condition_type_string == "adiabatic")  thermal_boundary_condition_type = ThermalBoundaryCondition::adiabatic;
        if (thermal_boundary_condition_type_string == "isothermal") thermal_boundary_condition_type = ThermalBoundaryCondition::isothermal;
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
