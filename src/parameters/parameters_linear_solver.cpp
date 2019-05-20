#include "parameters/parameters_linear_solver.h"

namespace Parameters
{
    using namespace dealii;
    // Linear solver inputs
    LinearSolverParam::LinearSolverParam () {}

    void LinearSolverParam::declare_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("linear solver");
        {
            prm.declare_entry("linear_solver_output", "quiet",
                              Patterns::Selection("quiet|verbose"),
                              "State whether output from linear solver should be printed. "
                              "Choices are <quiet|verbose>.");

            prm.declare_entry("linear_solver_type", "gmres",
                              Patterns::Selection("direct|gmres"),
                              "Enum of linear solver"
                              "Choices are <direct|gmres>.");

            prm.enter_subsection("gmres options");
            {
                prm.declare_entry("linear_residual_tolerance", "1e-4",
                                  Patterns::Double(),
                                  "Linear residual tolerance for convergence of the linear system");
                prm.declare_entry("max_iterations", "1000",
                                  Patterns::Integer(),
                                  "Maximum number of iterations for linear solver");

                // ILU with threshold parameters
                prm.declare_entry("ilut_fill", "2",
                                  Patterns::Integer(),
                                  "Amount of additional fill-in elements besides the sparse matrix structure");
                prm.declare_entry("ilut_drop", "1e-10",
                                  Patterns::Double(),
                                  "relative size of elements which should be dropped when forming an incomplete lu decomposition with threshold");
                prm.declare_entry("ilut_rtol", "1.1",
                                  Patterns::Double(),
                                  "Amount of an absolute perturbation that will be added to the diagonal of the matrix, "
                                  "which sometimes can help to get better preconditioners");
                prm.declare_entry("ilut_atol", "1e-9",
                                  Patterns::Double(),
                                  "Factor by which the diagonal of the matrix will be scaled, "
                                  "which sometimes can help to get better preconditioners");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
    }

    void LinearSolverParam ::parse_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("linear solver");
        {
            const std::string output_string = prm.get("linear_solver_output");
            if (output_string == "verbose") linear_solver_output = verbose;
            if (output_string == "quiet") linear_solver_output = quiet;

            const std::string solver_string = prm.get("linear_solver_type");
            if (solver_string == "direct") linear_solver_type = LinearSolverEnum::direct;

            if (solver_string == "gmres") 
            {
                linear_solver_type = LinearSolverEnum::gmres;
                prm.enter_subsection("gmres options");
                {
                    max_iterations  = prm.get_integer("max_iterations");
                    linear_residual = prm.get_double("linear_residual_tolerance");

                    ilut_fill = prm.get_integer("ilut_fill");
                    ilut_drop = prm.get_double("ilut_drop");
                    ilut_rtol = prm.get_double("ilut_rtol");
                    ilut_atol = prm.get_double("ilut_atol");
                }
                prm.leave_subsection();
            }
        }
        prm.leave_subsection();
    }
}
