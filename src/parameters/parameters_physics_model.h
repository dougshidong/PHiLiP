#ifndef __PARAMETERS_PHYSICS_MODEL_H__
#define __PARAMETERS_PHYSICS_MODEL_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to Physics Models
class PhysicsModelParam
{
public:
    /// Constructor
    PhysicsModelParam ();

    /// Types of physics models available.
    enum PhysicsModelEnum {
        large_eddy_simulation
    };
    /// Store the physics model type
    PhysicsModelEnum physics_model_type;

    /// Types of baseline physics available that the physics model can be built upon.
    enum BaselinePhysicsEnum {
        euler
       ,navier_stokes
    };
    /// Store the baseline physics type
    BaselinePhysicsEnum baseline_physics_type;

    /// Types of sub-grid scale (SGS) models that can be used.
    enum SubGridScaleModel { 
        smagorinsky
       ,wall_adaptive_local_eddy_viscosity
       ,dynamic_smagorinsky
    };
    /// Store the SubGridScale (SGS) model type
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
