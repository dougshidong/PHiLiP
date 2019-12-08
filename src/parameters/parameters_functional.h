#ifndef __PARAMETERS_FUNCTIONAL_H__
#define __PARAMETERS_FUNCTIONAL_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters_manufactured_solution.h"
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

class FunctionalParam
{
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
public:
    enum FunctionalType{
        normLp_volume,
        normLp_boundary,
        weighted_volume_integral,
        weighted_boundary_integral,
    };
    FunctionalType functional_type;

    double normLp;

    ManufacturedSolutionEnum weight_function_type;

    bool use_weight_function_laplacian;

    std::vector<unsigned int> boundary_vector;

    FunctionalParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_FUNCTIONAL_H__
