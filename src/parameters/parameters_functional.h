#ifndef __PARAMETERS_FUNCTIONAL_H__
#define __PARAMETERS_FUNCTIONAL_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters_manufactured_solution.h"
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameterse related to the functional object
class FunctionalParam
{
    /// Enumerator of manufactured solution types
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
public:
    /// Choices for functional types to be used
    enum FunctionalType{
        normLp_volume,
        normLp_boundary,
        weighted_integral_volume,
        weighted_integral_boundary,
        error_normLp_volume,
        error_normLp_boundary,
    };
    /// Selection of functinal type
    FunctionalType functional_type;

    /// Choice of Lp norm exponent used in functional calculation
    double normLp;

    /// Choice of manufactured solution function to be used in weighting expression
    ManufacturedSolutionEnum weight_function_type;

    /// Flag to use weight function laplacian
    bool use_weight_function_laplacian;

    /// Boundary of vector ids to be considered for boundary functional evaluation
    std::vector<unsigned int> boundary_vector;

    /// Flag for use of all domain boundaries
    bool use_all_boundaries;

    FunctionalParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_FUNCTIONAL_H__
