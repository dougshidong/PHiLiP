#include "parameters_flow_solver.h"

#include <string>

namespace PHiLiP {

namespace Parameters {

// Flow Solver inputs
FlowSolverParam::FlowSolverParam() {}

void FlowSolverParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("flow_solver");
    {
        prm.declare_entry("flow_case_type","taylor_green_vortex",
                          dealii::Patterns::Selection(
                          " taylor_green_vortex | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " periodic_1D_unsteady | "
                          " gaussian_bump | "
                          " sshock "),
                          "The type of flow we want to simulate. "
                          "Choices are "
                          " <taylor_green_vortex | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " periodic_1D_unsteady | "
                          " gaussian_bump | "
                          " sshock>. ");

        prm.declare_entry("poly_degree", "1",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Polynomial order (P) of the basis functions for DG.");

        prm.declare_entry("max_poly_degree_for_adaptation", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Maxiumum possible polynomial order (P) of the basis functions for DG "
                          "when doing adaptive simulations. Default is 0 which actually sets "
                          "the value to poly_degree in the code, indicating no adaptation.");

        prm.declare_entry("final_time", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Final solution time.");

        prm.declare_entry("courant_friedrich_lewy_number", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Courant-Friedrich-Lewy (CFL) number for constant time step.");

        prm.declare_entry("unsteady_data_table_filename", "unsteady_data_table",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Filename of the unsteady data table output file: unsteady_data_table_filename.txt.");

        prm.declare_entry("steady_state", "false",
                          dealii::Patterns::Bool(),
                          "Solve steady-state solution. False by default (i.e. unsteady by default).");

        prm.declare_entry("adaptive_time_step", "false",
                          dealii::Patterns::Bool(),
                          "Adapt the time step on the fly for unsteady flow simulations. False by default (i.e. constant time step by default).");

        prm.declare_entry("steady_state_polynomial_ramping", "false",
                          dealii::Patterns::Bool(),
                          "For steady-state cases, does polynomial ramping if set to true. False by default.");

        prm.declare_entry("sensitivity_table_filename", "sensitivity_table",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Filename for the sensitivity data table output file: sensitivity_table_filename.txt.");

        prm.declare_entry("restart_computation_from_file", "false",
                          dealii::Patterns::Bool(),
                          "Restarts the computation from the restart file. False by default.");

        prm.declare_entry("output_restart_files", "false",
                          dealii::Patterns::Bool(),
                          "Output restart files for restarting the computation. False by default.");

        prm.declare_entry("restart_files_directory_name", ".",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Name of directory for writing and reading restart files. Current directory by default.");

        prm.declare_entry("restart_file_index", "1",
                          dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                          "Index of restart file from which the computation will be restarted from. 1 by default.");

        prm.declare_entry("output_restart_files_every_x_steps", "1",
                          dealii::Patterns::Integer(1,dealii::Patterns::Integer::max_int_value),
                          "Outputs the restart files every x steps.");

        prm.declare_entry("output_restart_files_every_dt_time_intervals", "0.0",
                          dealii::Patterns::Double(0,dealii::Patterns::Double::max_double_value),
                          "Outputs the restart files at time intervals of dt.");

        prm.enter_subsection("grid");
        {
            prm.declare_entry("input_mesh_filename", "",
                              dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                              "Filename of the input mesh: input_mesh_filename.msh. For cases that import a mesh file.");

            prm.declare_entry("grid_degree", "1",
                              dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                              "Polynomial degree of the grid. Curvilinear grid if set greater than 1; default is 1.");

            prm.declare_entry("grid_left_bound", "0.0",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Left bound of domain for hyper_cube mesh based cases.");

            prm.declare_entry("grid_right_bound", "1.0",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Right bound of domain for hyper_cube mesh based cases.");

            prm.declare_entry("number_of_grid_elements_per_dimension", "4",
                              dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                              "Number of grid elements per dimension for hyper_cube mesh based cases.");

            prm.declare_entry("number_of_mesh_refinements", "0",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Number of mesh refinements for Gaussian bump and naca0012 based cases.");

            prm.declare_entry("use_input_mesh", "false",
                              dealii::Patterns::Bool(),
                              "Use the input .msh file which calls read_gmsh. False by default.");

            prm.declare_entry("mesh_reader_verbose_output", "false",
                              dealii::Patterns::Bool(),
                              "Flag for verbose (true) or quiet (false) mesh reader output.");

            prm.enter_subsection("boundary_IDs");
            {

                prm.declare_entry("use_periodic_BC_in_x", "false",
                                  dealii::Patterns::Bool(),
                                  "Use periodic boundary condition in the x-direction. False by default.");

                prm.declare_entry("use_periodic_BC_in_y", "false",
                                  dealii::Patterns::Bool(),
                                  "Use periodic boundary condition in the y-direction. False by default.");

                prm.declare_entry("use_periodic_BC_in_z", "false",
                                  dealii::Patterns::Bool(),
                                  "Use periodic boundary condition in the z-direction. False by default.");

                prm.declare_entry("x_periodic_id_face_1", "2001",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the first periodic boundary face in the x-direction.");

                prm.declare_entry("x_periodic_id_face_2", "2002",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the second periodic boundary face in the x-direction.");

                prm.declare_entry("y_periodic_id_face_1", "2003",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the first periodic boundary face in the y-direction.");

                prm.declare_entry("y_periodic_id_face_2", "2004",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the second periodic boundary face in the y-direction.");

                prm.declare_entry("z_periodic_id_face_1", "2005",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the first periodic boundary face in the z-direction.");

                prm.declare_entry("z_periodic_id_face_2", "2006",
                                  dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                  "Boundary ID for the second periodic boundary face in the z-direction.");
            }
            prm.leave_subsection();

            prm.enter_subsection("gaussian_bump");
            {
                prm.declare_entry("channel_length", "3.0",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Lenght of channel for gaussian bump meshes.");

                prm.declare_entry("channel_height", "0.8",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Height of channel for gaussian bump meshes.");

                prm.declare_entry("bump_height", "0.0625", 
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

        prm.enter_subsection("taylor_green_vortex");
        {
            prm.declare_entry("expected_kinetic_energy_at_final_time", "1",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "For integration test purposes, expected kinetic energy at final time.");

            prm.declare_entry("expected_theoretical_dissipation_rate_at_final_time", "1",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "For integration test purposes, expected theoretical kinetic energy dissipation rate at final time.");

            prm.declare_entry("density_initial_condition_type", "uniform",
                              dealii::Patterns::Selection(
                              " uniform | "
                              " isothermal "),
                              "The type of density initialization. "
                              "Choices are "
                              " <uniform | "
                              " isothermal>.");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void FlowSolverParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    prm.enter_subsection("flow_solver");
    {
        const std::string flow_case_type_string = prm.get("flow_case_type");
        if      (flow_case_type_string == "taylor_green_vortex")                {flow_case_type = taylor_green_vortex;}
        else if (flow_case_type_string == "burgers_viscous_snapshot")           {flow_case_type = burgers_viscous_snapshot;}
        else if (flow_case_type_string == "burgers_rewienski_snapshot")         {flow_case_type = burgers_rewienski_snapshot;}
        else if (flow_case_type_string == "naca0012")                           {flow_case_type = naca0012;}
        else if (flow_case_type_string == "periodic_1D_unsteady")                 {flow_case_type = periodic_1D_unsteady;}
        else if (flow_case_type_string == "gaussian_bump")                      {flow_case_type = gaussian_bump;}
        else if (flow_case_type_string == "sshock")                             {flow_case_type = sshock;}

        poly_degree = prm.get_integer("poly_degree");
        
        // get max poly degree for adaptation
        max_poly_degree_for_adaptation = prm.get_integer("max_poly_degree_for_adaptation");
        // -- set value to poly_degree if it is the default value
        if(max_poly_degree_for_adaptation == 0) max_poly_degree_for_adaptation = poly_degree;
        
        final_time = prm.get_double("final_time");
        courant_friedrich_lewy_number = prm.get_double("courant_friedrich_lewy_number");
        unsteady_data_table_filename = prm.get("unsteady_data_table_filename");
        steady_state = prm.get_bool("steady_state");
        steady_state_polynomial_ramping = prm.get_bool("steady_state_polynomial_ramping");
        adaptive_time_step = prm.get_bool("adaptive_time_step");
        sensitivity_table_filename = prm.get("sensitivity_table_filename");
        restart_computation_from_file = prm.get_bool("restart_computation_from_file");
        output_restart_files = prm.get_bool("output_restart_files");
        restart_files_directory_name = prm.get("restart_files_directory_name");
        restart_file_index = prm.get_integer("restart_file_index");
        output_restart_files_every_x_steps = prm.get_integer("output_restart_files_every_x_steps");
        output_restart_files_every_dt_time_intervals = prm.get_double("output_restart_files_every_dt_time_intervals");

        prm.enter_subsection("grid");
        {
            input_mesh_filename = prm.get("input_mesh_filename");
            grid_degree = prm.get_integer("grid_degree");
            grid_left_bound = prm.get_double("grid_left_bound");
            grid_right_bound = prm.get_double("grid_right_bound");
            number_of_grid_elements_per_dimension = prm.get_integer("number_of_grid_elements_per_dimension");
            number_of_mesh_refinements = prm.get_integer("number_of_mesh_refinements");
            use_input_mesh = prm.get_bool("use_input_mesh");
            mesh_reader_verbose_output = prm.get_bool("mesh_reader_verbose_output");

            prm.enter_subsection("boundary_IDs");
            {
                use_periodic_BC_in_x = prm.get_bool("use_periodic_BC_in_x");
                use_periodic_BC_in_y = prm.get_bool("use_periodic_BC_in_y");
                use_periodic_BC_in_z = prm.get_bool("use_periodic_BC_in_z");
                x_periodic_id_face_1 = prm.get_integer("x_periodic_id_face_1");
                x_periodic_id_face_2 = prm.get_integer("x_periodic_id_face_2");
                y_periodic_id_face_1 = prm.get_integer("y_periodic_id_face_1");
                y_periodic_id_face_2 = prm.get_integer("y_periodic_id_face_2");
                z_periodic_id_face_1 = prm.get_integer("z_periodic_id_face_1");
                z_periodic_id_face_2 = prm.get_integer("z_periodic_id_face_2");
            }
            prm.leave_subsection();

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

        prm.enter_subsection("taylor_green_vortex");
        {
            expected_kinetic_energy_at_final_time = prm.get_double("expected_kinetic_energy_at_final_time");
            expected_theoretical_dissipation_rate_at_final_time = prm.get_double("expected_theoretical_dissipation_rate_at_final_time");

            const std::string density_initial_condition_type_string = prm.get("density_initial_condition_type");
            if      (density_initial_condition_type_string == "uniform")    {density_initial_condition_type = uniform;}
            else if (density_initial_condition_type_string == "isothermal") {density_initial_condition_type = isothermal;}
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
