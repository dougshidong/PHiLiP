#ifndef __PARAMETERS_OPTIMIZATION_H__
#define __PARAMETERS_OPTIMIZATION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the linear solver
class OptimizationParameters
{
public:

    /// Types constrained optimization algorithms
    enum ConstrainedOptimizationAlgorithm {
        reduced_space_primal_dual_active_set,
        full_space_primal_dual_active_set,
    }

    /// Types unconstrained optimization algorithms
    enum UnconstrainedOptimizationAlgorithm {
        full_space,                ///< Full-space method from Biros and Ghattas.
        full_space_composite_step, ///< Full-space method from ROL.
        reduced_space_bfgs,        ///< Reduced-space quasi-Newton with BFGS approximation.
        reduced_space_newton,      ///< Reduced-space Newton-Krylov.
    };

    OptimizationAlgorithm optimization_algorithm; ///< Optimization algorithm.

    /// Identity regularization
    bool regularization_add_identity;
    double regularization_identity_multiplier_scaling;
    double regularization_identity_exponent_scaling;
    double regularization_identity_maximum;
    double regularization_identity_minimum;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);


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
};

} // Parameters namespace
} // PHiLiP namespace
#endif

