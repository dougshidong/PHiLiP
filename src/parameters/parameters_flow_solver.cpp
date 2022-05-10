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
                          " advection | "
                          " convection_diffusion"),
                          "The type of flow we want to simulate. "
                          "Choices are "
                          " <taylor_green_vortex | "
                          " burgers_viscous_snapshot | "
                          " naca0012 | "
                          " burgers_rewienski_snapshot | "
                          " burgers_inviscid | "
                          " advection | "
                          " convection_diffusion>.");

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
                          "Solve steady-state solution. False by default.");

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
      
        prm.declare_entry("input_mesh_filename", "naca0012",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Filename of the input mesh: input_mesh_filename.msh");

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
                          "Interpolate the initial condition function onto the DG solution. If fale, then it projects the physical value. True by default.");
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
        else if (flow_case_type_string == "advection")                  {flow_case_type = advection;}
        else if (flow_case_type_string == "convection_diffusion")       {flow_case_type = convection_diffusion;}

        final_time = prm.get_double("final_time");
        courant_friedrich_lewy_number = prm.get_double("courant_friedrich_lewy_number");
        unsteady_data_table_filename = prm.get("unsteady_data_table_filename");
        steady_state = prm.get_bool("steady_state");
        sensitivity_table_filename = prm.get("sensitivity_table_filename");
        restart_computation_from_file = prm.get_bool("restart_computation_from_file");
        output_restart_files = prm.get_bool("output_restart_files");
        restart_files_directory_name = prm.get("restart_files_directory_name");
        restart_file_index = prm.get_integer("restart_file_index");
        output_restart_files_every_x_steps = prm.get_integer("output_restart_files_every_x_steps");
        output_restart_files_every_dt_time_intervals = prm.get_double("output_restart_files_every_dt_time_intervals");
        input_mesh_filename = prm.get("input_mesh_filename");

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
    }
    prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
