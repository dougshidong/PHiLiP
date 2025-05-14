#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class ReducedOrderModelParam
{
public:

    /// Types of linear solvers available.
    enum LinearSolverEnum {
        direct, /// LU.
        gmres   /// GMRES.
    };

    /// Tolerance for POD adaptation
    double adaptation_tolerance;

    /// Path to search for snapshots or saved POD basis
    std::string path_to_search;

    /// Tolerance of the reduced-order nonlinear residual
    double reduced_residual_tolerance;

    /// Number of Halton sequence points to add to initial snapshot set
    int num_halton;

    /// Path to search for file with pre-determined snapshot locations used to build POD (actual FOM snapshots not calculated in advance) (should contain snapshot_table)
    std::string file_path_for_snapshot_locations;

    /// Recomputation parameter for adaptive sampling algorithm
    int recomputation_coefficient;

    /// Names of parameters
    std::vector<std::string> parameter_names;

    /// Minimum value of parameters
    std::vector<double> parameter_min_values;

    /// Maximum value of parameters
    std::vector<double> parameter_max_values;

    /// Type of linear solver used for first adjoint problem (DWR between FOM and ROM) (direct or gmres)
    LinearSolverEnum FOM_error_linear_solver_type; 

    /// Use residual/reduced residual for error indicator instead of DWR. False by default.
    bool residual_error_bool; 

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif