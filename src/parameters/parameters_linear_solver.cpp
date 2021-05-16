#include "parameters/parameters_linear_solver.h"

namespace PHiLiP {
namespace Parameters {

// Linear solver inputs
LinearSolverParam::LinearSolverParam () {}

void LinearSolverParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("linear solver");
    {
        prm.declare_entry("linear_solver_output", "quiet",
                          dealii::Patterns::Selection("quiet|verbose"),
                          "State whether output from linear solver should be printed. "
                          "Choices are <quiet|verbose>.");

        prm.declare_entry("linear_solver_type", "gmres",
                          dealii::Patterns::Selection("direct|gmres"),
                          "Enum of linear solver"
                          "Choices are <direct|gmres>.");

        prm.enter_subsection("gmres options");
        {
            prm.declare_entry("linear_residual_tolerance", "1e-4",
                              dealii::Patterns::Double(),
                              "Linear residual tolerance for convergence of the linear system");
            prm.declare_entry("max_iterations", "1000",
                              dealii::Patterns::Integer(),
                              "Maximum number of iterations for linear solver");
            prm.declare_entry("restart_number", "30",
                              dealii::Patterns::Integer(),
                              "Number of iterations before restarting GMRES");

            // ILU with threshold parameters
            prm.declare_entry("ilut_fill", "1",
                              dealii::Patterns::Integer(),
                              "Amount of additional fill-in elements besides the sparse matrix structure."
                              "For ilut_fill >= 1.0, "
                              "Number of entries to keep in the strict upper triangle of the "
                              " current row, and in the strict lower triangle of the current "
                              " row.  It does NOT correspond to the $p$ parameter in Saad's original. "
                              " description. This parameter represents a maximum fill fraction. "
                              " In this implementation, the L and U factors always contains nonzeros corresponding "
                              " to the original sparsity pattern of A, so this value should be >= 1.0. "
                              " Letting $fill = \frac{(level-of-fill - 1)*nnz(A)}{2*N}$, "
                              " each row of the computed L and U factors contains at most $fill$ "
                              " nonzero elements in addition to those from the sparsity pattern of A."
                              " For ilut_fill >= 1.0, "
                              " Typical graph-based level of-fill of the factorization such that the pattern corresponds to A^(p+1). ");
            prm.declare_entry("ilut_drop", "0.0",
                              dealii::Patterns::Double(),
                              "relative size of elements which should be dropped when forming an incomplete lu decomposition with threshold");
            prm.declare_entry("ilut_rtol", "1.0",
                              dealii::Patterns::Double(),
                              "Amount of an absolute perturbation that will be added to the diagonal of the matrix, "
                              "which sometimes can help to get better preconditioners");
            prm.declare_entry("ilut_atol", "0.0",
                              dealii::Patterns::Double(),
                              "Factor by which the diagonal of the matrix will be scaled, "
                              "which sometimes can help to get better preconditioners");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void LinearSolverParam ::parse_parameters (dealii::ParameterHandler &prm)
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
                restart_number  = prm.get_integer("restart_number");
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

} // Parameters namespace
} // PHiLiP namespace
