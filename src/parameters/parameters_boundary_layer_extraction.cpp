#include "parameters/parameters_boundary_layer_extraction.h"

namespace PHiLiP {
namespace Parameters {

// boundary layer extraction inputs
BoundaryLayerExtractionParam::BoundaryLayerExtractionParam () {}

void BoundaryLayerExtractionParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("boundary_layer_extraction");
    {
        prm.declare_entry("number_of_sampling", "100",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of sampling points on the extraction line. Default value is 100.");
        prm.declare_entry("extraction_point_x", "1.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The x coordinate of extraction start point. Default value is 1.0. ");
        prm.declare_entry("extraction_point_y", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The y coordinate of extraction start point. Default value is 0.0. ");
        prm.declare_entry("extraction_point_z", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The z coordinate of extraction start point. Default value is 0.0. ");
    }
    prm.leave_subsection();
}

void BoundaryLayerExtractionParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("boundary_layer_extraction");
    {
        number_of_sampling = prm.get_integer("number_of_sampling");
        extraction_point_x = prm.get_double("extraction_point_x");
        extraction_point_y = prm.get_double("extraction_point_y");
        extraction_point_z = prm.get_double("extraction_point_z");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace