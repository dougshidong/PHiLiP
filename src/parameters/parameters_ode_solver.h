#ifndef __PARAMETERS_ODE_SOLVER_H__
#define __PARAMETERS_ODE_SOLVER_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {
namespace Parameters {

/// Parameters related to the ODE solver.
class ODESolverParam
{
public:
    ODESolverParam (); ///< Constructor.
    /// Types of ODE solver
    enum ODESolverEnum {
        explicit_solver, /// RK4
        implicit_solver  /// Backward-Euler
    };

    OutputEnum ode_output; ///< verbose or quiet.
    ODESolverEnum ode_solver_type; ///< ODE solver type. Note that only implicit has been fully tested for now.

    unsigned int output_solution_every_x_steps; ///< Outputs the solution every x steps to .vtk file

    unsigned int nonlinear_max_iterations; ///< Maximum number of iterations.
    unsigned int print_iteration_modulo; ///< If ode_output==verbose, print every print_iteration_modulo iterations.

    double nonlinear_steady_residual_tolerance; ///< Tolerance to determine steady-state convergence.

    double initial_time_step; ///< Time step used in ODE solver.
    double time_step_factor_residual; ///< Multiplies initial time-step by time_step_factor_residual*(-log10(residual_norm_decrease))
    double time_step_factor_residual_exp; ///< Scales initial time step by pow(time_step_factor_residual*(-log10(residual_norm_decrease)),time_step_factor_residual_exp)

    static void declare_parameters (dealii::ParameterHandler &prm); ///< Declares the possible variables and sets the defaults.
    void parse_parameters (dealii::ParameterHandler &prm); ///< Parses input file and sets the variables.
};

} // Parameters namespace
} // PHiLiP namespace
#endif
