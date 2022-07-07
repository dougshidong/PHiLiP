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
                          " burgers_inviscid | "
                          " convection_diffusion | "
                          " advection | "
                          " advection_periodic"),
                          "The type of flow we want to simulate. "
                          "Choices are "
                          " <taylor_green_vortex | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " burgers_inviscid | "
                          " convection_diffusion | "
                          " advection | "
                          " advection_periodic>.");

        prm.declare_entry("poly_degree", "1",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Polynomial order (P) of the basis functions for DG.");

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
          prm.declare_entry("input_mesh_filename", "naca0012",
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

        prm.declare_entry("interpolate_initial_condition", "true",
                          dealii::Patterns::Bool(),
                          "Interpolate the initial condition function onto the DG solution. If false, then it projects the physical value. True by default.");
        
        prm.enter_subsection("time_refinement_study");
        {
            prm.declare_entry("number_of_times_to_solve", "4",
                              dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                              "Number of times to run the flow solver during a time refinement study.");
            prm.declare_entry("refinement_ratio", "0.5",
                              dealii::Patterns::Double(0, 1.0),
                              "Ratio between a timestep size and the next in a time refinement study, 0<r<1.");
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
        if      (flow_case_type_string == "taylor_green_vortex")        {flow_case_type = taylor_green_vortex;}
        else if (flow_case_type_string == "burgers_viscous_snapshot")   {flow_case_type = burgers_viscous_snapshot;}
        else if (flow_case_type_string == "burgers_rewienski_snapshot") {flow_case_type = burgers_rewienski_snapshot;}
        else if (flow_case_type_string == "naca0012")                   {flow_case_type = naca0012;}
        else if (flow_case_type_string == "burgers_inviscid")           {flow_case_type = burgers_inviscid;}
        else if (flow_case_type_string == "convection_diffusion")       {flow_case_type = convection_diffusion;}
        else if (flow_case_type_string == "advection")                  {flow_case_type = advection;}
        else if (flow_case_type_string == "advection_periodic")         {flow_case_type = advection_periodic;}

        poly_degree = prm.get_integer("poly_degree");
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

        interpolate_initial_condition = prm.get_bool("interpolate_initial_condition");

        prm.enter_subsection("time_refinement_study");
        {
            number_of_times_to_solve = prm.get_integer("number_of_times_to_solve");
            refinement_ratio = prm.get_double("refinement_ratio");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
