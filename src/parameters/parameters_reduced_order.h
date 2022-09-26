#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class ReducedOrderModelParam
{
public:

    /// Tolerance for POD adaptation
    double adaptation_tolerance;

    /// Path to search for snapshots or saved POD basis
    std::string path_to_search;

    /// Tolerance of the reduced-order nonlinear residual
    double reduced_residual_tolerance;

    /// Number of Halton sequence points to add to initial snapshot set
    int num_halton;

    /// Recomputation parameter for adaptive sampling algorithm
    int recomputation_coefficient;

    /// Names of parameters
    std::vector<std::string> parameter_names;

    /// Minimum value of parameters
    std::vector<double> parameter_min_values;

    /// Maximum value of parameters
    std::vector<double> parameter_max_values;

    ReducedOrderModelParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif