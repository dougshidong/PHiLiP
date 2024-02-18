#include "parameters/parameters_linear_solver.h"

namespace PHiLiP {
namespace Parameters {

void OptimizationParameters::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("optimization");
    {
        prm.declare_entry("max_design_cycles", "1000",
                          dealii::Patterns::Integer(),
                          "Maximum number of optimization design cycles, aka KKT iterations.");

        prm.declare_entry("optimization_algorithm", "reduced_space_bfgs",
                          dealii::Patterns::Selection(
                          "full_space|full_space_composite_step|reduced_space_bfgs|reduced_space_newton"
                          "Enum of linear solver"
                          "Choices are <full_space|full_space_composite_step|reduced_space_bfgs|reduced_space_newton>")

        prm.enter_subsection("full-space options");
        {
            prm.declare_entry("preconditioner_use_second_order", "false",
                              dealii::Patterns::Bool(),
                              "Use second-order terms in the preconditioner, aka P4, if true, and P2 if false."
                              "Results in 4 Jacobian inverses per preconditioner application versus 2."
                              "Choices are <true|false>");

            prm.declare_entry("preconditioner_use_approx_jacobian", "true",
                              dealii::Patterns::Bool(),
                              "Use Jacobian preconditioner instead of exact Jacobian within the KKT preconditioner."
                              "Choices are <true|false>");


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

        prm.enter_subsection("free-form deformation");
        {
            prm.declare_entry("nx_ffd", "10",
                              dealii::Patterns::Integer(),
                              "Number of FFD points in the X-direction..");
            prm.declare_entry("ny_ffd", "2",
                              dealii::Patterns::Integer(),
                              "Number of FFD points in the Y-direction..");
            prm.declare_entry("nz_ffd", "10",
                              dealii::Patterns::Integer(),
                              "Number of FFD points in the Z-direction..");

            prm.declare_entry("constrain_x", "true",
                              dealii::Patterns::Bool(),
                              "Remove FFD x-coordinates from design variables.");
                              "Choices are <true|false>");

            const unsigned int min_elements = 0;
            const unsigned int max_elements = 3;
            const std::string  separator=",";
            prm.declare_entry("constrain_xyz", "true, false, true",
                              dealii::Patterns::List (dealii::Patterns::Bool(), min_elements, max_elements, separator)
                              "Remove FFD x/y/z-coordinates from design variables.");
                              "Choices are < true|false, true|false, true|false > ");

            prm.declare_entry("constrain_y", "false",
                              dealii::Patterns::Bool(),
                              "Remove FFD y-coordinates from design variables.");
                              "Choices are <true|false>");
            prm.declare_entry("constrain_z", "true",
                              dealii::Patterns::Bool(),
                              "Remove FFD z-coordinates from design variables.");
                              "Choices are <true|false>");

            prm.declare_entry("constrain_leading_edge", "true",
                              dealii::Patterns::Bool(),
                              "Remove the first layer of FFD points in the x-direction from design variables.");
                              "Choices are <true|false>");
            prm.declare_entry("constrain_trailing_edge", "true",
                              dealii::Patterns::Bool(),
                              "Remove the last layer of FFD points in the x-direction from design variables.");
                              "Choices are <true|false>");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void OptimizationParameters ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("optimization");
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
