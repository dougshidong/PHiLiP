#ifndef __PARAMETERS_GRID_REFINEMENT_H__
#define __PARAMETERS_GRID_REFINEMENT_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

class GridRefinementParam
{
public:
    // main set of parameters for deciding the method

    enum RefinementMethod{
        uniform,        // all cells are refined
        fixed_fraction, // picks fraction with largest indicators
        continuous,     // generates a new mesh based on a size field
        };
    RefinementMethod refinement_method;

    enum RefinementType{
        h,  // element size only
        p,  // polynomial orders
        hp, // mix of both
        };
    RefinementType refinement_type;
    
    bool isotropic;

    enum ErrorIndicator{
        error_based,    // using the exact error for testing
        hessian_based,  // feature_based
        residual_based, // requires a fine grid projection
        adjoint_based,  // adjoint based
        };
    ErrorIndicator error_indicator;    

    // need to add: isotropy indicators AND smoothness indicator

    // double p; // polynomial order when fixed, should take this from the grid
    double q; // for the Lq norm

    // double fixed_fraction;

    GridRefinementParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_GRID_REFINEMENT_H__
