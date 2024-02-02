#ifndef __PARAMETERS_OPTIMIZATION_H__
#define __PARAMETERS_OPTIMIZATION_H__
 
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

/// Parameters for Optimization
class OptimizationParam 
{
public:
    /// Choices for optimization types. Can be reduced or full space.
    enum OptimizationType{
        reduced_space,
        full_space
    };
    /// Selection of type of optimization.
    OptimizationType optimization_type;
    
    /// Maximum number of optimization cycles (i.e. non-linear solves).
    int max_design_cycles;
    
    /// Maximum number of linear iterations to solve Ax=b.
    int linear_iteration_limit;

    /// Tolerance of the gradient at which we declare that the optimal values have been attained.
    double gradient_tolerance;

    /// Limit for number of functional evaluations during backtracking.
    int functional_evaluation_limit;

    /// Initial step size from which backtracking begins.
    double initial_step_size;
    
    /// Weight of mesh distortion added to the objective function to prevent mesh distortion during optimization.
    double mesh_weight_factor;

    /// Power to which cell volume is raised is raised.
    int mesh_volume_power;

    /// Preconditioner to be used with full space KKT system.
    std::string full_space_preconditioner;

    /// Line search curvature used to find step length (default is Strong Wolfe conditions).
    std::string line_search_curvature;

    /// Type of line search to be used (set to backtracking by default).
    std::string line_search_method;

    /// Descent method for reduced space. Can be Newton-Krylov or Quasi-Newton.
    std::string reduced_space_descent_method;

    /// Flag to subtract coarse residual from dual weighted residual. Used for full space goal oriented mesh optimization.
    bool use_coarse_residual;
    
    /// Flag to use fine solution as an initial guess. Used for full space goal oriented mesh optimization.
    bool use_fine_solution;

    /// Regularization parameter to be multiplied with identity and added to the hessian of control variables.
    double regularization_parameter_control;
    
    /// Regularization parameter to be multiplied with identity and added to the hessian of control variables.
    double regularization_parameter_sim;

    /// Scaling for regularization parameter after each iteration.
    double regularization_scaling;

    /// Regularization tol low.
    double regularization_tol_low;
    
    /// Regularization tol low.
    double regularization_tol_high;

    /// Constructor of mesh adaptation parameters.
    OptimizationParam();

    /// Declare parameters
    static void declare_parameters (dealii::ParameterHandler &prm);
 
    /// Parse parameters
    void parse_parameters (dealii::ParameterHandler &prm);

}; // class ends

} // namespace Parameters
} // namespace PHiLiP
#endif
