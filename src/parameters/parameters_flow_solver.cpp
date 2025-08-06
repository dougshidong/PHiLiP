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
                          " channel_flow | "
                          " isentropic_vortex | "
                          " kelvin_helmholtz_instability | "
                          " non_periodic_cube_flow | "
                          " sod_shock_tube | "
                          " low_density_2d | "
                          " leblanc_shock_tube | "
                          " shu_osher_problem | "
                          " advection_limiter | "
                          " burgers_limiter |"
                          " sshock | "
                          " wall_distance_evaluation | "
                          " flat_plate_2D | "
                          " airfoil_2D | "
                          " naca0012_turbulence "),
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
                          " channel_flow | "
                          " isentropic_vortex | "
                          " kelvin_helmholtz_instability | "
                          " non_periodic_cube_flow | "
                          " sod_shock_tube | "
                          " low_density_2d | "
                          " leblanc_shock_tube | "
                          " shu_osher_problem | "
                          " advection_limiter | "
                          " burgers_limiter >. |"
                          " sshock | "
                          " wall_distance_evaluation | "
                          " flat_plate_2D | "
                          " airfoil_2D | "
                          " naca0012_turbulence ");

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
        
        prm.declare_entry("write_unsteady_data_table_file_every_dt_time_intervals", "0.0",
                          dealii::Patterns::Double(0,dealii::Patterns::Double::max_double_value),
                          "Writes the unsteady data table file at time intervals of dt. "
                          "If set to zero, it outputs at every time step.");

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

            prm.enter_subsection("flat_plate_2D");
            {
                prm.declare_entry("free_length", "0.5",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Lenght of free area upwind to the flat plate.");

                prm.declare_entry("free_height", "1.0",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Height of free area above of the flat plate.");

                prm.declare_entry("plate_length", "2.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Lenght of the flat plate.");

                prm.declare_entry("skewness_x_free", "1.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Skewness of the meshes in the x direction.");

                prm.declare_entry("skewness_x_plate", "1.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Skewness of the meshes in the x direction.");

                prm.declare_entry("skewness_y", "1.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Skewness of the meshes in the y direction.");

                prm.declare_entry("skewness_z", "1.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Skewness of the meshes in the z direction.");

                prm.declare_entry("number_of_subdivisions_in_x_direction_free", "0",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions in the x direction of free area for flat plate meshes.");

                prm.declare_entry("number_of_subdivisions_in_x_direction_plate", "0",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions in the x direction of plate area for flat plate meshes.");

                prm.declare_entry("number_of_subdivisions_in_y_direction", "0",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions in the y direction for flat plate meshes.");

                prm.declare_entry("number_of_subdivisions_in_z_direction", "0",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions in the z direction for flat plate meshes.");
            }
            prm.leave_subsection();

            prm.enter_subsection("airfoil_2D");
            {
                prm.declare_entry("airfoil_length", "1.0",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Lenght of airfoil.");

                prm.declare_entry("height", "2.0",
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Height of area surround airfoil.");

                prm.declare_entry("length_b2", "2.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Lenght between trailing edge and outlet farfield.");

                prm.declare_entry("incline_factor", "0.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Inclination factor.");

                prm.declare_entry("bias_factor", "1.0", 
                                  dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                                  "Bias factor.");

                prm.declare_entry("refinements", "0", 
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of global refinements.");

                prm.declare_entry("n_subdivision_x_0", "30", 
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions along the airfoil in left block.");

                prm.declare_entry("n_subdivision_x_1", "10",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions along the airfoil in middle block.");

                prm.declare_entry("n_subdivision_x_2", "30",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions along the airfoil in right block.");

                prm.declare_entry("n_subdivision_y", "20",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Number of subdivisions normal to the airfoil contour.");

                prm.declare_entry("airfoil_sampling_factor", "3",
                                  dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                                  "Airfoil sampling factor.");
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
            prm.declare_entry("check_nonphysical_flow_case_behavior", "false",
                              dealii::Patterns::Bool(),
                              "Flag to check if non-physical case dependant behaviour is encounted. By default, false.");
        }
        prm.leave_subsection();

        prm.enter_subsection("channel_flow");
        {
            prm.declare_entry("channel_friction_velocity_reynolds_number", "590",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Channel Reynolds number based on wall friction velocity. Default is 590.");

            prm.declare_entry("turbulent_channel_number_of_cells_x_direction","4",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Number of cells in the x-direction for channel flow case.");

            prm.declare_entry("turbulent_channel_number_of_cells_y_direction","16",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Number of cells in the y-direction for channel flow case.");

            prm.declare_entry("turbulent_channel_number_of_cells_z_direction","2",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Number of cells in the z-direction for channel flow case.");

            prm.declare_entry("turbulent_channel_domain_length_x_direction", "6.283185307179586476",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Channel domain length for x-direction. Default is 2*PI.");

            prm.declare_entry("turbulent_channel_domain_length_y_direction", "2.0",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Channel domain length for y-direction. Default is 2.0.");

            prm.declare_entry("turbulent_channel_domain_length_z_direction", "3.141592653589793238",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Channel domain length for x-direction. Default is PI.");

            prm.declare_entry("turbulent_channel_mesh_stretching_function_type", "gullbrand",
                              dealii::Patterns::Selection(
                              " gullbrand | "
                              " carton_de_wiart_et_al | "
                              " hopw "),
                              "The type of mesh stretching function for channel flow case. "
                              "Choices are "
                              " <gullbrand | "
                              " carton_de_wiart_et_al | "
                              " hopw>.");
            prm.declare_entry("xvelocity_initial_condition_type", "laminar",
                              dealii::Patterns::Selection(
                              " laminar | "
                              " manufactured | "
                              " turbulent "),
                              "The type of x-velocity initialization. "
                              "Choices are "
                              " <laminar | "
                              " manufactured | "
                              " turbulent>.");
            prm.declare_entry("relaxation_coefficient_for_turbulent_channel_flow_source_term", "0.0",
                              dealii::Patterns::Double(-dealii::Patterns::Double::max_double_value, dealii::Patterns::Double::max_double_value),
                              "Relaxation coefficient for the turbulent channel flow source term. Default is 0.");
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

            prm.declare_entry("output_density_field_in_addition_to_velocity", "false",
                              dealii::Patterns::Bool(),
                              "Output density field in addition to velocity field. False by default.");
            prm.declare_entry("output_viscosity_field_in_addition_to_velocity", "false",
                              dealii::Patterns::Bool(),
                              "Output viscosity field in addition to velocity field. False by default.");

            prm.declare_entry("compute_time_averaged_solution", "false",
                              dealii::Patterns::Bool(),
                              "Compute time averaged solution on the fly (for example, to get velocity fluctuations). False by default.");
            prm.declare_entry("time_to_start_averaging", "0.0",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Time after which the time avering of the solution starts. 0.0 default.");
            prm.declare_entry("compute_Reynolds_stress", "false",
                              dealii::Patterns::Bool(),
                              "Compute time averaged solution on the fly (for example, to get velocity fluctuations). False by default.");
            prm.declare_entry("time_to_start_computing_Reynolds_Stress", "0.0",
                              dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                              "Time after which the time avering of the solution starts. 0.0 default.");            
            prm.declare_entry("output_flow_field_files_directory_name", ".",
                              dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                              "Name of directory for writing flow field files. Current directory by default.");
        }
        prm.leave_subsection();

        prm.declare_entry("end_exactly_at_final_time", "true",
                          dealii::Patterns::Bool(),
                          "Flag to adjust the last timestep such that the simulation "
                          "ends exactly at final_time. True by default.");

        prm.declare_entry("do_compute_unsteady_data_and_write_to_table", "true",
                          dealii::Patterns::Bool(),
                          "Flag for computing unsteady data and writting to table. True by default.");
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
        else if (flow_case_type_string == "channel_flow")               {flow_case_type = channel_flow;}
        else if (flow_case_type_string == "isentropic_vortex")          {flow_case_type = isentropic_vortex;}
        else if (flow_case_type_string == "kelvin_helmholtz_instability")   
                                                                        {flow_case_type = kelvin_helmholtz_instability;}
        else if (flow_case_type_string == "non_periodic_cube_flow")     {flow_case_type = non_periodic_cube_flow;}
        else if (flow_case_type_string == "sod_shock_tube")             {flow_case_type = sod_shock_tube;}
        else if (flow_case_type_string == "low_density_2d")             {flow_case_type = low_density_2d;}
        else if (flow_case_type_string == "leblanc_shock_tube")         {flow_case_type = leblanc_shock_tube;}
        else if (flow_case_type_string == "shu_osher_problem")          {flow_case_type = shu_osher_problem;}
        else if (flow_case_type_string == "advection_limiter")          {flow_case_type = advection_limiter;}
        else if (flow_case_type_string == "burgers_limiter")            {flow_case_type = burgers_limiter;}
        
        else if (flow_case_type_string == "sshock")                     {flow_case_type = sshock;}
        else if (flow_case_type_string == "wall_distance_evaluation")   {flow_case_type = wall_distance_evaluation;}
        else if (flow_case_type_string == "flat_plate_2D")              {flow_case_type = flat_plate_2D;}
        else if (flow_case_type_string == "airfoil_2D")                 {flow_case_type = airfoil_2D;}
        else if (flow_case_type_string == "naca0012_turbulence")                   {flow_case_type = naca0012_turbulence;}

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
        write_unsteady_data_table_file_every_dt_time_intervals = prm.get_double("write_unsteady_data_table_file_every_dt_time_intervals");

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

            // prm.enter_subsection("flat_plate_2D");
            // {
            //     number_of_subdivisions_in_x_direction_free = prm.get_integer("number_of_subdivisions_in_x_direction_free");
            //     number_of_subdivisions_in_x_direction_plate = prm.get_integer("number_of_subdivisions_in_x_direction_plate");
            //     number_of_subdivisions_in_y_direction = prm.get_integer("number_of_subdivisions_in_y_direction");
            //     number_of_subdivisions_in_z_direction = prm.get_integer("number_of_subdivisions_in_z_direction");
            //     free_length = prm.get_double("free_length");
            //     free_height = prm.get_double("free_height");
            //     plate_length = prm.get_double("plate_length");
            //     skewness_x_free = prm.get_double("skewness_x_free");
            //     skewness_x_plate = prm.get_double("skewness_x_plate");
            //     skewness_y = prm.get_double("skewness_y");
            //     skewness_z = prm.get_double("skewness_z");
            // }
            // prm.leave_subsection();

            prm.enter_subsection("airfoil_2D");
            {
                airfoil_length = prm.get_double("airfoil_length");
                height = prm.get_double("height");
                length_b2 = prm.get_double("length_b2");
                incline_factor = prm.get_double("incline_factor");
                bias_factor = prm.get_double("bias_factor");
                refinements = prm.get_integer("refinements");
                n_subdivision_x_0 = prm.get_integer("n_subdivision_x_0");
                n_subdivision_x_1 = prm.get_integer("n_subdivision_x_1");
                n_subdivision_x_2 = prm.get_integer("n_subdivision_x_2");
                n_subdivision_y = prm.get_integer("n_subdivision_y");
                airfoil_sampling_factor = prm.get_integer("airfoil_sampling_factor");
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
            check_nonphysical_flow_case_behavior = prm.get_bool("check_nonphysical_flow_case_behavior");
        }
        prm.leave_subsection();

        prm.enter_subsection("channel_flow");
        {
            turbulent_channel_friction_velocity_reynolds_number = prm.get_double("channel_friction_velocity_reynolds_number");
            turbulent_channel_number_of_cells_x_direction = prm.get_integer("turbulent_channel_number_of_cells_x_direction");
            turbulent_channel_number_of_cells_y_direction = prm.get_integer("turbulent_channel_number_of_cells_y_direction");
            turbulent_channel_number_of_cells_z_direction = prm.get_integer("turbulent_channel_number_of_cells_z_direction");
            turbulent_channel_domain_length_x_direction = prm.get_double("turbulent_channel_domain_length_x_direction");
            turbulent_channel_domain_length_y_direction = prm.get_double("turbulent_channel_domain_length_y_direction");
            turbulent_channel_domain_length_z_direction = prm.get_double("turbulent_channel_domain_length_z_direction");
            const std::string turbulent_channel_mesh_stretching_function_type_string = prm.get("turbulent_channel_mesh_stretching_function_type");
            if      (turbulent_channel_mesh_stretching_function_type_string == "gullbrand")             {turbulent_channel_mesh_stretching_function_type = gullbrand;}
            else if (turbulent_channel_mesh_stretching_function_type_string == "hopw")                  {turbulent_channel_mesh_stretching_function_type = hopw;}
            else if (turbulent_channel_mesh_stretching_function_type_string == "carton_de_wiart_et_al") {turbulent_channel_mesh_stretching_function_type = carton_de_wiart_et_al;}
            const std::string xvelocity_initial_condition_type_string = prm.get("xvelocity_initial_condition_type");
            if      (xvelocity_initial_condition_type_string == "laminar")      {xvelocity_initial_condition_type = laminar;}
            else if (xvelocity_initial_condition_type_string == "turbulent")    {xvelocity_initial_condition_type = turbulent;}
            else if (xvelocity_initial_condition_type_string == "manufactured") {xvelocity_initial_condition_type = manufactured;}
            relaxation_coefficient_for_turbulent_channel_flow_source_term = prm.get_double("relaxation_coefficient_for_turbulent_channel_flow_source_term");
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
          output_density_field_in_addition_to_velocity = prm.get_bool("output_density_field_in_addition_to_velocity");
          output_viscosity_field_in_addition_to_velocity = prm.get_bool("output_viscosity_field_in_addition_to_velocity");
          compute_time_averaged_solution = prm.get_bool("compute_time_averaged_solution");
          time_to_start_averaging = prm.get_double("time_to_start_averaging");
          compute_Reynolds_stress = prm.get_bool("compute_Reynolds_stress");
          time_to_start_computing_Reynolds_Stress = prm.get_double("time_to_start_computing_Reynolds_Stress");
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
        do_compute_unsteady_data_and_write_to_table = prm.get_bool("do_compute_unsteady_data_and_write_to_table");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
