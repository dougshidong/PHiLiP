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
    /// Types of ODE solver
    enum ODESolverEnum {
        runge_kutta_solver, /// Runge-Kutta (RK), explicit or diagonally implicit 
        implicit_solver,  /// Backward-Euler
        rrk_explicit_solver, /// Explicit RK using the relaxation Runge-Kutta method (Ketcheson, 2019)
        pod_galerkin_solver, ///Proper Orthogonal Decomposition with Galerkin projection
        pod_petrov_galerkin_solver ///Proper Orthogonal Decomposition with Petrov-Galerkin projection (LSPG)
    };

    OutputEnum ode_output; ///< verbose or quiet.
    ODESolverEnum ode_solver_type; ///< ODE solver type.

    int output_solution_every_x_steps; ///< Outputs the solution every x steps to .vtk file
    double output_solution_every_dt_time_intervals; ///< Outputs the solution every dt time intervals to .vtk file
    bool output_solution_at_fixed_times; ///< Flag for outputting solution at fixed times
    std::string output_solution_fixed_times_string; ///< String of fixed solution output times
    unsigned int number_of_fixed_times_to_output_solution; ///< Number of fixed times to output the solution
    bool output_solution_at_exact_fixed_times; ///< Flag for outputting the solution at exact fixed times by decreasing the time step on the fly

    unsigned int nonlinear_max_iterations; ///< Maximum number of iterations.
    unsigned int print_iteration_modulo; ///< If ode_output==verbose, print every print_iteration_modulo iterations.
    bool output_final_steady_state_solution_to_file; ///< Output final steady state solution to file
    std::string steady_state_final_solution_filename; ///< Filename to write final steady state solution

    double nonlinear_steady_residual_tolerance; ///< Tolerance to determine steady-state convergence.

    double initial_time_step; ///< Time step used in ODE solver.
    double time_step_factor_residual; ///< Multiplies initial time-step by time_step_factor_residual*(-log10(residual_norm_decrease))
    double time_step_factor_residual_exp; ///< Scales initial time step by pow(time_step_factor_residual*(-log10(residual_norm_decrease)),time_step_factor_residual_exp)

    /** Set as false by default. 
      * If true, writes the linear solver convergence data for
      *  steady state to a file named "ode_solver_steady_state_convergence_data_table.txt"
      */
    bool output_ode_solver_steady_state_convergence_table;
  
    double initial_time; ///< Initial time at which we initialize the ODE solver with.
    unsigned int initial_iteration; ///< Initial iteration at which we initialize the ODE solver with.
    /** Initial desired time for outputting the solution every dt time intervals 
        at which we initialize the ODE solver with. */
    double initial_desired_time_for_output_solution_every_dt_time_intervals;

    /// Types of RK method (i.e., unique Butcher tableau)
    enum RKMethodEnum {
        rk4_ex, ///Classical fourth-order RK
        ssprk3_ex, ///Third-order strong-stability preserving
        heun2_ex, ///Heun's method (second order explicit trapezoidal; SSP)
        euler_ex, ///Forward Euler
        euler_im, ///Implicit Euler
        dirk_2_im, ///Second-order diagonally-implicit RK
        dirk_3_im ///Third-order diagonally-implicit RK
    };

    RKMethodEnum runge_kutta_method; ///< Runge-kutta method.
    int n_rk_stages; ///< Number of stages for an RK method; assigned based on runge_kutta_method
    int rk_order; ///< Order of the RK method; assigned based on runge_kutta_method

    /// Flag to signal that automatic differentiation (AD) matrix dRdW must be allocated
    bool allocate_matrix_dRdW;

    /// Do output for root solving routine
    OutputEnum rrk_root_solver_output;

    /// Tolerance for RRK root solver, default value 5E-10
    double relaxation_runge_kutta_root_tolerance;

    static void declare_parameters (dealii::ParameterHandler &prm); ///< Declares the possible variables and sets the defaults.
    void parse_parameters (dealii::ParameterHandler &prm); ///< Parses input file and sets the variables.
};

} // Parameters namespace
} // PHiLiP namespace
#endif
