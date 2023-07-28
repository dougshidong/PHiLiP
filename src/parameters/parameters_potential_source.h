#ifndef __PARAMETERS_POTENTIAL_SOURCE_H__
#define __PARAMETERS_POTENTIAL_SOURCE_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

/// Parameters related to the source geometry
class PotentialSourceParam
{
public:
    // Solution parameters
    bool use_viscous_drag; ///> applies viscous drag contribution to the physical source 

    // Selects the potential source geometry used
    enum PotentialSourceGeometry{
        trailing_edge_serrations,
        circular_test,
        none,
        };
    PotentialSourceGeometry potential_source_geometry; ///< Selected PotentialSourceGeometry from the input file

    /// Trailing Edge Serrations (TES) options
    double TES_frequency; ///< frequency of Trailing Edge Serrations.
    double TES_h; ///< half length of Trailing Edge Serrations. 
    double TES_thickness; ///< thickness of Trailing Edge Serrations.
    //// to achieve significant noise reduction 2h >= freq [Gruber, 2012] ////

    double TES_effective_length_factor; ///< effective length factor -> experimentally determined, dependent on airfoil
    /// Input file provides in degrees, but the value stored here is in radians
    double TES_flap_angle; ///< angle between Trailing Edge Serration and chord.

    /// Circular Test options
    double circle_radius; 

    PotentialSourceParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif