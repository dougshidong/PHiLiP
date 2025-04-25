#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include "parameters_flow_solver.h"

#include <string>

//for checking output directories
#include <sys/types.h>
#include <sys/stat.h>

namespace PHiLiP {

namespace Parameters {

void FlowSolverParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("flow_solver");
    {
        prm.declare_entry("flow_case_type","taylor_green_vortex",
                          dealii::Patterns::Selection(
                          " taylor_green_vortex | "
                          " decaying_homogeneous_isotropic_turbulence | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " burgers_inviscid | "
                          " convection_diffusion | "
                          " advection | "
                          " periodic_1D_unsteady | "
                          " gaussian_bump | "
                          " isentropic_vortex | "
                          " kelvin_helmholtz_instability | "
                          " non_periodic_cube_flow | "
                          " sod_shock_tube | "
                          " low_density | "
                          " leblanc_shock_tube | "
                          " shu_osher_problem | "
                          " advection_limiter | "
                          " burgers_limiter | "
                          " double_mach_reflection | "
                          " shock_diffraction | "
                          " astrophysical_jet | "
                          " strong_vortex_shock_wave |"),
                          "The type of flow we want to simulate. "
                          "Choices are "
                          " <taylor_green_vortex | "
                          " decaying_homogeneous_isotropic_turbulence | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " burgers_inviscid | "
                          " convection_diffusion | "
                          " advection | "
                          " periodic_1D_unsteady | "
                          " gaussian_bump | "
                          " isentropic_vortex | "
                          " kelvin_helmholtz_instability | "
                          " non_periodic_cube_flow | "
                          " sod_shock_tube | "
                          " low_density | "
                          " leblanc_shock_tube | "
                          " shu_osher_problem | "
                          " advection_limiter | "
                          " burgers_limiter | "
                          " double_mach_reflection | "
                          " shock_diffraction | "
                          " astrophysical_jet | "
                          " strong_vortex_shock_wave >. ");

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

        prm.declare_entry("constant_time_step", "0",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Constant time step.");

        prm.declare_entry("courant_friedrichs_lewy_number", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Courant-Friedrich-Lewy (CFL) number for constant time step.");

        prm.declare_entry("unsteady_data_table_filename", "unsteady_data_table",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Filename of the unsteady data table output file: unsteady_data_table_filename.txt.");

        prm.declare_entry("steady_state", "false",
                          dealii::Patterns::Bool(),
                          "Solve steady-state solution. False by default (i.e. unsteady by default).");

        prm.declare_entry("error_adaptive_time_step", "false",
                          dealii::Patterns::Bool(),
                          "Adapt the time step on the fly for unsteady flow simulations according to an estimate of temporal error. False by default (i.e. constant time step by default).");

        prm.declare_entry("adaptive_time_step", "false",
                          dealii::Patterns::Bool(),
                          "Adapt the time step on the fly for unsteady flow simulations according to a CFL condition. False by default (i.e. constant time step by default).");

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

        prm.declare_entry("expected_order_at_final_time", "0.0",
                  dealii::Patterns::Double(0.0, 10.0),
                  "For convergence tests related to limiters, expected order of accuracy for final run.");

        prm.enter_subsection("grid");
        {
            prm.declare_entry("input_mesh_filename", "",
                              dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                              "Filename of the input mesh: input_mesh_filename.msh. For cases that import a mesh file.");

            prm.declare_entry("grid_degree", "1",
                              dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                              "Polynomial degree of the grid. Curvilinear grid if set greater than 1; default is 1.");

            prm.declare_entry("grid_left_bound", "0.0",
                              dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                              "Left bound of domain for hyper_cube mesh based cases.");

            prm.declare_entry("grid_right_bound", "1.0",
                              dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                              "Right bound of domain for hyper_cube mesh based cases.");

            prm.declare_entry("number_of_grid_elements_per_dimension", "4",
                              dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                              "Number of grid elements per dimension for hyper_cube mesh based cases.");

            prm.declare_entry("number_of_mesh_refinements", "0",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Number of mesh refinements for Gaussian bump and naca0012 based cases.");

            prm.declare_entry("use_gmsh_mesh", "false",
                              dealii::Patterns::Bool(),
                              "Use the input .msh file which calls read_gmsh. False by default.");

            prm.declare_entry("mesh_reader_verbose_output", "false",
                              dealii::Patterns::Bool(),
                              "Flag for verbose (true) or quiet (false) mesh reader output.");

            prm.enter_subsection("gmsh_boundary_IDs");
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

          prm.enter_subsection("positivity_preserving_tests");
          {
              prm.declare_entry("grid_xmin", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Left bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("grid_xmax", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Right bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("grid_ymin", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Left bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("grid_ymax", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Right bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("grid_zmin", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Left bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("grid_zmax", "0.0",
                                dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                                "Right bound of domain for hyper_cube mesh based cases.");

              prm.declare_entry("number_of_grid_elements_x", "1",
                                dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                "Number of grid elements in the x-direction for 2/3D positivity-preserving limiter cases.");

              prm.declare_entry("number_of_grid_elements_y", "1",
                                dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                "Number of grid elements in the y-direction for 2/3D positivity-preserving limiter cases.");

              prm.declare_entry("number_of_grid_elements_z", "1",
                                dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                                "Number of grid elements in the z-direction for 2/3D positivity-preserving limiter cases.");
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
            
            prm.declare_entry("do_calculate_numerical_entropy", "false",
                              dealii::Patterns::Bool(),
                              "Flag to calculate numerical entropy and write to file. By default, do not calculate.");
            
        }
        prm.leave_subsection();

        prm.enter_subsection("kelvin_helmholtz_instability");
        {
            prm.declare_entry("atwood_number", "0.5",
                              dealii::Patterns::Double(0.0, 1.0),
                              "Atwood number, which characterizes the density difference "
                              "between the layers of fluid.");
        }
        prm.leave_subsection();

        prm.declare_entry("apply_initial_condition_method", "interpolate_initial_condition_function",
                          dealii::Patterns::Selection(
                          " interpolate_initial_condition_function | "
                          " project_initial_condition_function | "
                          " read_values_from_file_and_project "),
                          "The method used for applying the initial condition. "
                          "Choices are "
                          " <interpolate_initial_condition_function | "
                          " project_initial_condition_function | " 
                          " read_values_from_file_and_project>.");

        prm.declare_entry("input_flow_setup_filename_prefix", "setup",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Filename prefix of the input flow setup file. "
                          "Example: 'setup' for files named setup-0000i.dat, where i is the MPI rank. "
                          "For initializing the flow with values from a file. "
                          "To be set when apply_initial_condition_method is read_values_from_file_and_project.");

        prm.enter_subsection("output_velocity_field");
        {
            prm.declare_entry("output_velocity_field_at_fixed_times", "false",
                              dealii::Patterns::Bool(),
                              "Output velocity field (at equidistant nodes) at fixed times. False by default.");

            prm.declare_entry("output_velocity_field_times_string", " ",
                              dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                              "String of the times at which to output the velocity field. "
                              "Example: '0.0 1.0 2.0 3.0 ' or '0.0 1.0 2.0 3.0'");

            prm.declare_entry("output_vorticity_magnitude_field_in_addition_to_velocity", "false",
                              dealii::Patterns::Bool(),
                              "Output vorticity magnitude field in addition to velocity field. False by default.");

            prm.declare_entry("output_flow_field_files_directory_name", ".",
                              dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                              "Name of directory for writing flow field files. Current directory by default.");
        }
        prm.leave_subsection();

        prm.declare_entry("end_exactly_at_final_time", "true",
                          dealii::Patterns::Bool(),
                          "Flag to adjust the last timestep such that the simulation "
                          "ends exactly at final_time. True by default.");
    }
    prm.leave_subsection();
}

void FlowSolverParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    prm.enter_subsection("flow_solver");
    {
        const std::string flow_case_type_string = prm.get("flow_case_type");
        if      (flow_case_type_string == "taylor_green_vortex")        {flow_case_type = taylor_green_vortex;}
        else if (flow_case_type_string == "decaying_homogeneous_isotropic_turbulence") 
                                                                        {flow_case_type = decaying_homogeneous_isotropic_turbulence;}
        else if (flow_case_type_string == "burgers_viscous_snapshot")   {flow_case_type = burgers_viscous_snapshot;}
        else if (flow_case_type_string == "burgers_rewienski_snapshot") {flow_case_type = burgers_rewienski_snapshot;}
        else if (flow_case_type_string == "naca0012")                   {flow_case_type = naca0012;}
        else if (flow_case_type_string == "burgers_inviscid")           {flow_case_type = burgers_inviscid;}
        else if (flow_case_type_string == "convection_diffusion")       {flow_case_type = convection_diffusion;}
        else if (flow_case_type_string == "advection")                  {flow_case_type = advection;}
        else if (flow_case_type_string == "periodic_1D_unsteady")       {flow_case_type = periodic_1D_unsteady;}
        else if (flow_case_type_string == "gaussian_bump")              {flow_case_type = gaussian_bump;}
        else if (flow_case_type_string == "isentropic_vortex")          {flow_case_type = isentropic_vortex;}
        else if (flow_case_type_string == "kelvin_helmholtz_instability")   
                                                                        {flow_case_type = kelvin_helmholtz_instability;}
        else if (flow_case_type_string == "non_periodic_cube_flow")     {flow_case_type = non_periodic_cube_flow;}
        // Positivity Preserving Tests
        else if (flow_case_type_string == "sod_shock_tube")             {flow_case_type = sod_shock_tube;}
        else if (flow_case_type_string == "low_density")                {flow_case_type = low_density;}
        else if (flow_case_type_string == "leblanc_shock_tube")         {flow_case_type = leblanc_shock_tube;}
        else if (flow_case_type_string == "shu_osher_problem")          {flow_case_type = shu_osher_problem;}
        else if (flow_case_type_string == "advection_limiter")          {flow_case_type = advection_limiter;}
        else if (flow_case_type_string == "burgers_limiter")            {flow_case_type = burgers_limiter;}
        else if (flow_case_type_string == "double_mach_reflection")     {flow_case_type = double_mach_reflection;}
        else if (flow_case_type_string == "shock_diffraction")          {flow_case_type = shock_diffraction;}
        else if (flow_case_type_string == "astrophysical_jet")          {flow_case_type = astrophysical_jet;}
        else if (flow_case_type_string == "strong_vortex_shock_wave")   {flow_case_type = strong_vortex_shock_wave;}
        
        poly_degree = prm.get_integer("poly_degree");
        
        // get max poly degree for adaptation
        max_poly_degree_for_adaptation = prm.get_integer("max_poly_degree_for_adaptation");
        // -- set value to poly_degree if it is the default value
        if(max_poly_degree_for_adaptation == 0) max_poly_degree_for_adaptation = poly_degree;
        final_time = prm.get_double("final_time");
        constant_time_step = prm.get_double("constant_time_step");
        courant_friedrichs_lewy_number = prm.get_double("courant_friedrichs_lewy_number");
        unsteady_data_table_filename = prm.get("unsteady_data_table_filename");
        steady_state = prm.get_bool("steady_state");
        steady_state_polynomial_ramping = prm.get_bool("steady_state_polynomial_ramping");
        error_adaptive_time_step = prm.get_bool("error_adaptive_time_step");
        adaptive_time_step = prm.get_bool("adaptive_time_step");
        sensitivity_table_filename = prm.get("sensitivity_table_filename");
        restart_computation_from_file = prm.get_bool("restart_computation_from_file");
        output_restart_files = prm.get_bool("output_restart_files");
        restart_files_directory_name = prm.get("restart_files_directory_name");
        // Check if directory exists - see https://stackoverflow.com/a/18101042
        struct stat info_restart;
        if( stat( restart_files_directory_name.c_str(), &info_restart ) != 0 ){
            pcout << "Error: No restart files directory named " << restart_files_directory_name << " exists." << std::endl
                      << "Please create the directory and restart. Aborting..." << std::endl;
            std::abort();
        }
        restart_file_index = prm.get_integer("restart_file_index");
        output_restart_files_every_x_steps = prm.get_integer("output_restart_files_every_x_steps");
        output_restart_files_every_dt_time_intervals = prm.get_double("output_restart_files_every_dt_time_intervals");
        expected_order_at_final_time = prm.get_double("expected_order_at_final_time");

        prm.enter_subsection("grid");
        {
            input_mesh_filename = prm.get("input_mesh_filename");
            grid_degree = prm.get_integer("grid_degree");
            grid_left_bound = prm.get_double("grid_left_bound");
            grid_right_bound = prm.get_double("grid_right_bound");
            number_of_grid_elements_per_dimension = prm.get_integer("number_of_grid_elements_per_dimension");
            number_of_mesh_refinements = prm.get_integer("number_of_mesh_refinements");
            use_gmsh_mesh = prm.get_bool("use_gmsh_mesh");
            mesh_reader_verbose_output = prm.get_bool("mesh_reader_verbose_output");

            prm.enter_subsection("gmsh_boundary_IDs");
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

            prm.enter_subsection("positivity_preserving_tests");
            {
                grid_xmax = prm.get_double("grid_xmax");
                grid_xmin = prm.get_double("grid_xmin");
                grid_ymax = prm.get_double("grid_ymax");
                grid_ymin = prm.get_double("grid_ymin");
                grid_zmax = prm.get_double("grid_zmax");
                grid_zmin = prm.get_double("grid_zmin");

                number_of_grid_elements_x = prm.get_integer("number_of_grid_elements_x");
                number_of_grid_elements_y = prm.get_integer("number_of_grid_elements_y");
                number_of_grid_elements_z = prm.get_integer("number_of_grid_elements_z");
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
            do_calculate_numerical_entropy = prm.get_bool("do_calculate_numerical_entropy");
        }
        prm.leave_subsection();

        prm.enter_subsection("kelvin_helmholtz_instability");
        {
            atwood_number = prm.get_double("atwood_number");
        }
        prm.leave_subsection();

        const std::string apply_initial_condition_method_string = prm.get("apply_initial_condition_method");
        if      (apply_initial_condition_method_string == "interpolate_initial_condition_function") {apply_initial_condition_method = interpolate_initial_condition_function;}
        else if (apply_initial_condition_method_string == "project_initial_condition_function")     {apply_initial_condition_method = project_initial_condition_function;}
        else if (apply_initial_condition_method_string == "read_values_from_file_and_project")      {apply_initial_condition_method = read_values_from_file_and_project;}
        
        input_flow_setup_filename_prefix = prm.get("input_flow_setup_filename_prefix");

        prm.enter_subsection("output_velocity_field");
        {
          output_velocity_field_at_fixed_times = prm.get_bool("output_velocity_field_at_fixed_times");
          output_velocity_field_times_string = prm.get("output_velocity_field_times_string");
          number_of_times_to_output_velocity_field = get_number_of_values_in_string(output_velocity_field_times_string);
          output_vorticity_magnitude_field_in_addition_to_velocity = prm.get_bool("output_vorticity_magnitude_field_in_addition_to_velocity");
          output_flow_field_files_directory_name = prm.get("output_flow_field_files_directory_name");
            // Check if directory exists - see https://stackoverflow.com/a/18101042
            struct stat info_flow;
            if( stat( output_flow_field_files_directory_name.c_str(), &info_flow ) != 0 ){
                pcout << "Error: No flow field files directory named " << output_flow_field_files_directory_name << " exists." << std::endl
                          << "Please create the directory and restart. Aborting..." << std::endl;
                std::abort();
            }
        }
        prm.leave_subsection();

        end_exactly_at_final_time = prm.get_bool("end_exactly_at_final_time");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
