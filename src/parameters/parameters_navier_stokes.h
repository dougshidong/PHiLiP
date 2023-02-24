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
    double temperature_inf; ///< Farfield temperature in degree Kelvin [K]
    double nondimensionalized_isothermal_wall_temperature; ///< Nondimensionalized isothermal wall temperature

    /// Types of thermal boundary conditions available.
    enum ThermalBoundaryCondition {
        adiabatic,
        isothermal
    };
    ThermalBoundaryCondition thermal_boundary_condition_type; ///< Store thermal boundary condition type
    
    bool use_constant_viscosity; /// Flag for using constant viscosity
    double nondimensionalized_constant_viscosity; ///< Nondimensionalized constant viscosity value

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
