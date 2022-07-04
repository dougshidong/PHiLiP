#include "parameters_mesh_generation.h"

#include <string>

namespace PHiLiP {

namespace Parameters {

MeshGenerationParam::MeshGenerationParam() {}

void MeshGenerationParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh_generation");
    {
        prm.declare_entry("mesh_type","straight_periodic_cube",
                          dealii::Patterns::Selection(
                          " straight_periodic_cube | "
                          " curved_periodic_grid | "
                          " gaussian_bump "),
                          "The type of mesh we want to generate. "
                          "Choices are "
                          " <straight_periodic_cube | "
                          " curved_periodic_grid | "
                          " gaussian_bump>. ");

        prm.declare_entry("grid_degree", "1",
                            dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                            "Polynomial degree of the grid. Curvilinear grid if set greater than 1; default is 1.");

        prm.declare_entry("input_mesh_filename", "straight_periodic_cube",
                            dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                            "Filename of the input mesh: input_mesh_filename.msh. For cases that import a mesh file.");

        prm.enter_subsection("straight_periodic_cube");
        {
          prm.declare_entry("grid_left_bound", "0.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Left bound of domain for straight_periodic_cube mesh.");

          prm.declare_entry("grid_right_bound", "1.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Right bound of domain for straight_periodic_cube mesh.");

          prm.declare_entry("number_of_grid_elements_per_dimension", "4",
                            dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                            "Number of grid elements per dimension for straight_periodic_cube mesh.");
        }
        prm.leave_subsection();

        prm.enter_subsection("curved_periodic_grid");
        {
          prm.declare_entry("grid_left_bound", "0.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Left bound of domain for curved_periodic_grid mesh.");

          prm.declare_entry("grid_right_bound", "1.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Right bound of domain for curved_periodic_grid mesh.");

          prm.declare_entry("number_of_grid_elements_per_dimension", "4",
                            dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                            "Number of grid elements per dimension for curved_periodic_grid mesh.");
        }
        prm.leave_subsection();


        prm.enter_subsection("gaussian_bump");
        {
          prm.declare_entry("channel_length", "1.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Lenght of channel for gaussian bump meshes.");

          prm.declare_entry("channel_height", "1.0",
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Height of channel for gaussian bump meshes.");

          prm.declare_entry("bump_height", "0.1", 
                            dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                            "Height of the bump for gaussian bump meshes.");

          prm.declare_entry("number_of_subdivisions_in_x_direction", "0",
                            dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                            "Number of subdivisions in the x direction for gaussian bump meshes.");

          prm.declare_entry("number_of_subdivisions_in_y_direction", "0",
                            dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                            "Number of subdivisions in the y direction for gaussian bump meshes.");

          prm.declare_entry("number_of_subdivisions_in_z_direction", "0",
                            dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                            "Number of subdivisions in the z direction for gaussian bump meshes.");

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void MeshGenerationParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    prm.enter_subsection("mesh_generation");
    {
        const std::string mesh_type_string = prm.get("mesh_type");
        if      (mesh_type_string == "straight_periodic_cube")     {mesh_type = straight_periodic_cube;}
        else if (mesh_type_string == "curved_periodic_grid")       {mesh_type = curved_periodic_grid;}
        else if (mesh_type_string == "gaussian_bump")              {mesh_type = gaussian_bump;}

        input_mesh_filename = prm.get("input_mesh_filename");
        grid_degree = prm.get_integer("grid_degree");

        // default for hyper_cube based meshes
        prm.enter_subsection("straight_periodic_cube");
        {
          grid_left_bound = prm.get_double("grid_left_bound");
          grid_right_bound = prm.get_double("grid_right_bound");
          number_of_grid_elements_per_dimension = prm.get_integer("number_of_grid_elements_per_dimension");
        }
        prm.leave_subsection();

        // overwrite if curved period grid (hyper cube)
        if(mesh_type == curved_periodic_grid) {
          prm.enter_subsection("curved_periodic_grid");
          {
            grid_left_bound = prm.get_double("grid_left_bound");
            grid_right_bound = prm.get_double("grid_right_bound");
            number_of_grid_elements_per_dimension = prm.get_integer("number_of_grid_elements_per_dimension");
          }
          prm.leave_subsection();
        }

      // overwrite if gaussian bump mesh
        prm.enter_subsection("gaussian_bump");
        {
          number_of_subdivisions_in_x_direction = prm.get_integer("number_of_subdivisions_in_x_direction");
          number_of_subdivisions_in_y_direction = prm.get_integer("number_of_subdivisions_in_y_direction");
          number_of_subdivisions_in_z_direction = prm.get_integer("number_of_subdivisions_in_z_direction");
          channel_length = prm.get_double("channel_length");
          channel_height = prm.get_double("channel_height");
          bump_height = prm.get_double("bump_height");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
