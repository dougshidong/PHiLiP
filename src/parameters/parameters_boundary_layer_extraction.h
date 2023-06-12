#ifndef __PARAMETERS_BOUNDARY_LAYER_EXTRACTION_H__
#define __PARAMETERS_BOUNDARY_LAYER_EXTRACTION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to boundary layer extraction
class BoundaryLayerExtractionParam
{
public:
    int number_of_sampling; ///< The number of sampling points on the extraction line.
    double extraction_point_x; ///< The x coordinate of extraction start point.
    double extraction_point_y; ///< The y coordinate of extraction start point.
    double extraction_point_z; ///< The z coordinate of extraction start point.

    BoundaryLayerExtractionParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif