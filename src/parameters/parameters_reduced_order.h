#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class ReducedOrderModelParam
{
public:
    /// Parameter a for eq.(18) in Carlberg 2011
    double rewienski_a;

    /// Parameter b for eq.(18) in Carlberg 2011
    double rewienski_b;

    /// Initial dimension of the coarse basis
    unsigned int coarse_basis_dimension;

    /// Initial dimension of the fine basis
    unsigned int fine_basis_dimension;

    /// Tolerance for POD adaptation
    double adaptation_tolerance;

    /// Number of basis functions to add at each iteration of POD adaptation. Set to 0 to determine online.
    unsigned int adapt_coarse_basis_constant;

    /** Set as true for running a manufactured solution.
     *  Adds the manufactured solution source term to the PDE source term
     */
    bool rewienski_manufactured_solution;

    /// Path to search for snapshots or saved POD basis
    std::string path_to_search;

    /// Use the method of snapshots to compute POD basis
    bool method_of_snapshots;

    /// Consider the sign of the error estimate from the dual-weighted residual
    bool consider_error_sign;

    ReducedOrderModelParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif
