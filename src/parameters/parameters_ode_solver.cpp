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

        prm.declare_entry("ode_solver_type", "implicit",
                          dealii::Patterns::Selection("explicit|implicit"),
                          "Explicit or implicit solver"
                          "Choices are <explicit|implicit>.");

        prm.declare_entry("nonlinear_max_iterations", "500000",
                          dealii::Patterns::Integer(1,dealii::Patterns::Integer::max_int_value),
                          "Maximum nonlinear solver iterations");
        prm.declare_entry("nonlinear_steady_residual_tolerance", "1e-13",
                          dealii::Patterns::Double(1e-16,dealii::Patterns::Double::max_double_value),
                          "Nonlinear solver residual tolerance");
        prm.declare_entry("time_step", "100.0",
                          dealii::Patterns::Double(1e-16,dealii::Patterns::Double::max_double_value),
                          "Time step used in ODE solver.");

        prm.declare_entry("print_iteration_modulo", "1",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Print every print_iteration_modulo iterations of "
                          "the nonlinear solver");
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

        const std::string solver_string = prm.get("ode_solver_type");
        if (solver_string == "explicit") ode_solver_type = ODESolverEnum::explicit_solver;
        if (solver_string == "implicit") ode_solver_type = ODESolverEnum::implicit_solver;

        nonlinear_steady_residual_tolerance  = prm.get_double("nonlinear_steady_residual_tolerance");
        nonlinear_max_iterations = prm.get_integer("nonlinear_max_iterations");
        time_step  = prm.get_double("time_step");

        print_iteration_modulo = prm.get_integer("print_iteration_modulo");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
