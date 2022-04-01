#ifndef __PARAMETERS_NAVIER_STOKES_H__
#define __PARAMETERS_NAVIER_STOKES_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the Navier Stokes equations
class NavierStokesParam
{
public:
    NavierStokesParam (); ///< Constructor

    double prandtl_number; ///< Prandtl number
    double reynolds_number_inf; ///< Farfield Reynolds number
    double isothermal_wall_temperature; ///< Isothermal wall temperature

    /// Types of wall temperature boundary conditions available.
    enum WallTemperatureBoundaryConditionEnum {
        adiabatic, /// adiabatic
        isothermal /// isothermal
    };
    /// Wall temperature boundary condition type
    WallTemperatureBoundaryConditionEnum wall_temperature_boundary_condition_type; ///< adiabatic or isothermal

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
