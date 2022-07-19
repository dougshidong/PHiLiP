#include "parameters/parameters_ode_solver.h"

namespace PHiLiP {
namespace Parameters {

ODESolverParam::ODESolverParam () {}

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

        prm.declare_entry("ode_solver_type", "implicit",
                          dealii::Patterns::Selection("explicit|implicit|pod_galerkin|pod_petrov_galerkin"),
                          "Explicit or implicit solver, or reduced-order POD Galerkin or POD Petrov Galerkin solver"
                          "Choices are <explicit|implicit|pod_galerkin|pod_petrov_galerkin>.");

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

        prm.enter_subsection("explicit solver options");
        {
            prm.declare_entry("runge_kutta_order", "3",
                              dealii::Patterns::Selection("1|3|4"),
                              "Order for the Runge-Kutta explicit time advancement scheme."
                              "Choices are <1|3|4>.");
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
        if (output_string == "verbose") ode_output = OutputEnum::verbose;

        output_solution_every_x_steps = prm.get_integer("output_solution_every_x_steps");
        output_solution_every_dt_time_intervals = prm.get_double("output_solution_every_dt_time_intervals");

        const std::string solver_string = prm.get("ode_solver_type");
        if (solver_string == "explicit") ode_solver_type = ODESolverEnum::explicit_solver;
        if (solver_string == "implicit") ode_solver_type = ODESolverEnum::implicit_solver;
        if (solver_string == "pod_galerkin") ode_solver_type = ODESolverEnum::pod_galerkin_solver;
        if (solver_string == "pod_petrov_galerkin") ode_solver_type = ODESolverEnum::pod_petrov_galerkin_solver;

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

        prm.enter_subsection("explicit solver options");
        {
            const std::string runge_kutta_order_string = prm.get("runge_kutta_order");
            if (runge_kutta_order_string == "1") runge_kutta_order = 1;
            if (runge_kutta_order_string == "3") runge_kutta_order = 3;
            if (runge_kutta_order_string == "4") runge_kutta_order = 4;
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
