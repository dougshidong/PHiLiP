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
        prm.declare_entry("output_solution_vector_modulo", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Output solution vector every output_solution_vector_modulo iterations of "
                          "the nonlinear solver. Set to 0 to disable.");
        prm.declare_entry("solutions_table_filename", "solutions_table",
                          dealii::Patterns::Anything(),
                          "Filename to use when outputting solution vectors in a table format.");

        prm.enter_subsection("explicit solver options");
        {
            prm.declare_entry("runge_kutta_order", "3",
                              dealii::Patterns::Selection("1|3"),
                              "Order for the Runge-Kutta explicit time advancement scheme."
                              "Choices are <1|3>.");
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
        output_solution_vector_modulo = prm.get_integer("output_solution_vector_modulo");
        solutions_table_filename = prm.get("solutions_table_filename");

        prm.enter_subsection("explicit solver options");
        {
            const std::string runge_kutta_order_string = prm.get("runge_kutta_order");
            if (runge_kutta_order_string == "1") runge_kutta_order = 1;
            if (runge_kutta_order_string == "3") runge_kutta_order = 3;
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
