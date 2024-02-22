#include "parameters/parameters_ode_solver.h"

namespace PHiLiP {
namespace Parameters {

void ODESolverParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("ODE solver");
    {
        prm.declare_entry("ode_output", "verbose",
                          dealii::Patterns::Selection("quiet|verbose"),
                          "State whether output from ODE solver should be printed. "
                          "Choices are <quiet|verbose>.");

        prm.declare_entry("output_solution_every_x_steps", "-1",
                          dealii::Patterns::Integer(-1,dealii::Patterns::Integer::max_int_value),
                          "Outputs the solution every x steps in .vtk file");

        prm.declare_entry("output_solution_every_dt_time_intervals", "0.0",
                          dealii::Patterns::Double(0,dealii::Patterns::Double::max_double_value),
                          "Outputs the solution at time intervals of dt in .vtk file");

        prm.declare_entry("output_solution_at_fixed_times", "false",
                          dealii::Patterns::Bool(),
                          "Output solution at fixed times. False by default.");

        prm.declare_entry("output_solution_fixed_times_string", " ",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "String of the times at which to output the velocity field. "
                          "Example: '0.0 1.0 2.0 3.0 ' or '0.0 1.0 2.0 3.0'");

        prm.declare_entry("output_solution_at_exact_fixed_times", "false",
                          dealii::Patterns::Bool(),
                          "Output solution at exact fixed times by decreasing the time step on the fly. False by default. "
                          "NOTE: Should be set to false if doing stability studies so that the time step is never influenced by solution file soutput times.");

        prm.declare_entry("ode_solver_type", "implicit",
                          dealii::Patterns::Selection(
                          " runge_kutta | "
                          " implicit | "
                          " rrk_explicit | "
                          " pod_galerkin | "
                          " pod_petrov_galerkin"),
                          "Type of ODE solver to use."
                          "Choices are "
                          " <runge_kutta | "
                          " implicit | "
                          " rrk_explicit | "
                          " pod_galerkin | "
                          " pod_petrov_galerkin>.");

        prm.declare_entry("nonlinear_max_iterations", "500000",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Maximum nonlinear solver iterations");
        prm.declare_entry("nonlinear_steady_residual_tolerance", "1e-13",
                          //dealii::Patterns::Double(1e-16,dealii::Patterns::Double::max_double_value),
                          dealii::Patterns::Double(1e-300,dealii::Patterns::Double::max_double_value),
                          "Nonlinear solver residual tolerance");
        prm.declare_entry("initial_time_step", "100.0",
                          dealii::Patterns::Double(1e-16,dealii::Patterns::Double::max_double_value),
                          "Time step used in ODE solver.");
        prm.declare_entry("time_step_factor_residual", "0.0",
                          dealii::Patterns::Double(0,dealii::Patterns::Double::max_double_value),
                          "Multiplies initial time-step by time_step_factor_residual*(-log10(residual_norm_decrease)).");
        prm.declare_entry("time_step_factor_residual_exp", "1.0",
                          dealii::Patterns::Double(0,dealii::Patterns::Double::max_double_value),
                          "Scales initial time step by pow(time_step_factor_residual*(-log10(residual_norm_decrease)),time_step_factor_residual_exp).");

        prm.declare_entry("print_iteration_modulo", "1",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Print every print_iteration_modulo iterations of "
                          "the nonlinear solver");
        prm.declare_entry("output_final_steady_state_solution_to_file", "false",
                          dealii::Patterns::Bool(),
                          "Output final steady state solution to file if set to true");
        prm.declare_entry("steady_state_final_solution_filename", "solution_snapshot",
                          dealii::Patterns::Anything(),
                          "Filename to use when outputting solution to a file.");
        prm.declare_entry("output_ode_solver_steady_state_convergence_table","false",
                          dealii::Patterns::Bool(),
                          "Set as false by default. If true, writes the linear solver convergence data "
                          "for steady state to a file named 'ode_solver_steady_state_convergence_data_table.txt'.");

        prm.declare_entry("initial_time", "0.0",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Initial time at which we initialize the ODE solver with.");

        prm.declare_entry("initial_iteration", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Initial iteration at which we initialize the ODE solver with.");
        
        prm.declare_entry("initial_desired_time_for_output_solution_every_dt_time_intervals", "0.0",
                           dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                           "Initial desired time for outputting the solution every dt time intervals "
                           "at which we initialize the ODE solver with.");

        prm.declare_entry("runge_kutta_method", "ssprk3_ex",
                          dealii::Patterns::Selection(
                          " rk4_ex | "
                          " ssprk3_ex | "
                          " heun2_ex | "
                          " euler_ex | "
                          " euler_im | "
                          " dirk_2_im | "
                          " dirk_3_im"),
                          "Runge-kutta method to use. Methods with _ex are explicit, and with _im are implicit."
                          "Choices are "
                          " <rk4_ex | "
                          " ssprk3_ex | "
                          " heun2_ex | "
                          " euler_ex | "
                          " euler_im | "
                          " dirk_2_im | "
                          " dirk_3_im>.");
        prm.enter_subsection("rrk root solver");
        {
            prm.declare_entry("rrk_root_solver_output", "quiet",
                              dealii::Patterns::Selection("quiet|verbose"),
                              "State whether output from rrk root solver should be printed. "
                              "Choices are <quiet|verbose>.");

            prm.declare_entry("relaxation_runge_kutta_root_tolerance", "5e-10",
                              dealii::Patterns::Double(),
                              "Tolerance for root-finding problem in entropy RRK ode solver."
                              "Defult 5E-10 is suitable in most cases.");
        }
        prm.leave_subsection();

    }
    prm.leave_subsection();
}

void ODESolverParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("ODE solver");
    {
        const std::string output_string = prm.get("ode_output");
        if (output_string == "quiet")   ode_output = OutputEnum::quiet;
        else if (output_string == "verbose") ode_output = OutputEnum::verbose;

        output_solution_every_x_steps = prm.get_integer("output_solution_every_x_steps");
        output_solution_every_dt_time_intervals = prm.get_double("output_solution_every_dt_time_intervals");
        output_solution_at_fixed_times = prm.get_bool("output_solution_at_fixed_times");
        output_solution_fixed_times_string = prm.get("output_solution_fixed_times_string");
        number_of_fixed_times_to_output_solution = get_number_of_values_in_string(output_solution_fixed_times_string);
        output_solution_at_exact_fixed_times = prm.get_bool("output_solution_at_exact_fixed_times");

        // Assign ode_solver_type and the allocate AD matrix dRdW flag
        const std::string solver_string = prm.get("ode_solver_type");
        if (solver_string == "runge_kutta")              { ode_solver_type = ODESolverEnum::runge_kutta_solver;
                                                           allocate_matrix_dRdW = false; }
        else if (solver_string == "implicit")            { ode_solver_type = ODESolverEnum::implicit_solver;
                                                           allocate_matrix_dRdW = true; }
        else if (solver_string == "rrk_explicit")        { ode_solver_type = ODESolverEnum::rrk_explicit_solver;
                                                           allocate_matrix_dRdW = false; }
        else if (solver_string == "pod_galerkin")        { ode_solver_type = ODESolverEnum::pod_galerkin_solver;
                                                           allocate_matrix_dRdW = true; }
        else if (solver_string == "pod_petrov_galerkin") { ode_solver_type = ODESolverEnum::pod_petrov_galerkin_solver;
                                                           allocate_matrix_dRdW = true; }

        nonlinear_steady_residual_tolerance  = prm.get_double("nonlinear_steady_residual_tolerance");
        nonlinear_max_iterations = prm.get_integer("nonlinear_max_iterations");
        initial_time_step  = prm.get_double("initial_time_step");
        time_step_factor_residual = prm.get_double("time_step_factor_residual");
        time_step_factor_residual_exp = prm.get_double("time_step_factor_residual_exp");

        print_iteration_modulo = prm.get_integer("print_iteration_modulo");
        output_final_steady_state_solution_to_file = prm.get_bool("output_final_steady_state_solution_to_file");
        steady_state_final_solution_filename = prm.get("steady_state_final_solution_filename");
        output_ode_solver_steady_state_convergence_table = prm.get_bool("output_ode_solver_steady_state_convergence_table");

        initial_time = prm.get_double("initial_time");
        initial_iteration = prm.get_integer("initial_iteration");
        initial_desired_time_for_output_solution_every_dt_time_intervals = prm.get_double("initial_desired_time_for_output_solution_every_dt_time_intervals");
        
        const std::string rk_method_string = prm.get("runge_kutta_method");
        if (rk_method_string == "rk4_ex"){
            runge_kutta_method = RKMethodEnum::rk4_ex;
            n_rk_stages  = 4;
            rk_order = 4;
        }
        else if (rk_method_string == "ssprk3_ex"){
            runge_kutta_method = RKMethodEnum::ssprk3_ex;
            n_rk_stages  = 3;
            rk_order = 3;
        }
        else if (rk_method_string == "heun2_ex"){
            runge_kutta_method = RKMethodEnum::heun2_ex;
            n_rk_stages  = 2;
            rk_order = 2;
        }
        else if (rk_method_string == "euler_ex"){
            runge_kutta_method = RKMethodEnum::euler_ex;
            n_rk_stages  = 1;
            rk_order = 1;
        }
        else if (rk_method_string == "euler_im"){
            runge_kutta_method = RKMethodEnum::euler_im;
            n_rk_stages  = 1;
            rk_order = 1;
        }
        else if (rk_method_string == "dirk_2_im"){
            runge_kutta_method = RKMethodEnum::dirk_2_im;
            n_rk_stages  = 2;
            rk_order = 2;
        }
        else if (rk_method_string == "dirk_3_im"){
            runge_kutta_method = RKMethodEnum::dirk_3_im;
            n_rk_stages  = 3;
            rk_order = 3;
        }
        prm.enter_subsection("rrk root solver");
        {
            const std::string output_string = prm.get("rrk_root_solver_output");
            if (output_string == "verbose") rrk_root_solver_output = verbose;
            if (output_string == "quiet") rrk_root_solver_output = quiet;

            relaxation_runge_kutta_root_tolerance = prm.get_double("relaxation_runge_kutta_root_tolerance");
        }
        prm.leave_subsection();

    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
