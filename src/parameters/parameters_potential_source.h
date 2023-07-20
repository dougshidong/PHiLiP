#ifndef __PARAMETERS_POTENTIAL_SOURCE_H__
#define __PARAMETERS_POTENTIAL_SOURCE_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

/// Parameters related to the source geometry
class PotentialSourceParam
{
public:
    // Trailing Edge Serrations (TES)
    //// to achieve significant noise reduction 2h >= freq [Gruber, 2012] ////
    double TES_frequency; ///< frequency of Trailing Edge Serrations.
    double TES_h; ///< half length of Trailing Edge Serrations. 
    double TES_thickness; ///< thickness of Trailing Edge Serrations.

    double TES_effective_length_factor; ///< effective length factor -> experimentally determined, dependent on airfoil
    /// Input file provides in degrees, but the value stored here is in radians
    double TES_gamma; ///< angle between Trailing Edge Serration and chord.

    // Solution parameters
    bool use_viscous_drag;

    PotentialSourceParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif