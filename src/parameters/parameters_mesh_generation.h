#ifndef __PARAMETERS_MESH_GENERATION_H__
#define __PARAMETERS_MESH_GENERATION_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameters related to generating meshes
class MeshGenerationParam
{
public:
    MeshGenerationParam(); ///< Constructor

    /// Selects the type of mesh to be generated
    enum MeshType{
        straight_periodic_cube,
        curved_periodic_grid,
        gaussian_bump
        };
    MeshType mesh_type; ///< Selected MeshType from the input file

    /** Name of the Gmsh file to be read if the case reads a mesh;
     *  will read file: input_mesh_filename.msh */
    std::string input_mesh_filename;
    unsigned int grid_degree; ///< Polynomial degree of the grid

    double grid_left_bound; ///< Left bound of domain for hyper_cube mesh based cases
    double grid_right_bound; ///< Right bound of domain for hyper_cube mesh based cases
    unsigned int number_of_grid_elements_per_dimension; ///< Number of grid elements per dimension for hyper_cube mesh based cases

    double channel_height; ///< Height of channel for gaussian bump case
    double channel_length; ///< Width of channel for gaussian bump case
    int number_of_subdivisions_in_x_direction; ///< Number of subdivisions in x direction for gaussian bump case
    int number_of_subdivisions_in_y_direction; ///< Number of subdivisions in y direction for gaussian bump case
    int number_of_subdivisions_in_z_direction; ///< Number of subdivisions in z direction for gaussian bump case

    /// Declares the possible variables and sets the defaults (default case is straight_periodic_cube).
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif

