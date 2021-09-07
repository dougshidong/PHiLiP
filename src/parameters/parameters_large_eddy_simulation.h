#ifndef __PARAMETERS_LARGE_EDDY_SIMULATION_H__
#define __PARAMETERS_LARGE_EDDY_SIMULATION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to Large Eddy Simulation
class LargeEddySimulationParam
{
public:
    /// Constructor
    LargeEddySimulationParam ();

    /// Types of sub-grid scale models that can be used.
    enum SubGridScaleModel { 
        smagorinsky
       ,wall_adaptive_local_eddy_viscosity
       ,dynamic_smagorinsky
    };

    /// SubGridScale (SGS) model type
    SubGridScaleModel SGS_model_type;

    double turbulent_prandtl_number; ///< Turbulent Prandtl number

    double smagorinsky_model_constant; ///< Smagorinsky Model Constant

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
