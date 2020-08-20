#ifndef __PARAMETERS_GRID_REFINEMENT_H__
#define __PARAMETERS_GRID_REFINEMENT_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

class GridRefinementParam
{
public:
    unsigned int refinement_steps;

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
    
    bool anisotropic;

    // maximum anisotropic ratio for continuous size field targets
    double anisotropic_ratio_max;

    // minimum anisotropic ratio for continuous zie field targets
    double anisotropic_ratio_min;

    // threshold value in anisotropic indicator to enable
    // anisotropic splitting (for fixed fraction methods)
    double anisotropic_threshold_ratio;

    enum AnisoIndicator{
        jump_based,           // based on face jumps with neighbouring cells
        reconstruction_based, // based on p+1 derivative along tensor product chord lines
        };
    AnisoIndicator anisotropic_indicator;

    enum ErrorIndicator{
        error_based,    // using the exact error for testing
        hessian_based,  // feature_based
        residual_based, // requires a fine grid projection
        adjoint_based,  // adjoint based
        };
    ErrorIndicator error_indicator;    

    // file type/interface to use
    enum OutputType{
        gmsh_out, // output of pos and geo files for gmsh remeshing
        msh_out,  // output of .msh with data fields corresponding to output_data_type
        };
    OutputType output_type;

    // method of data storage in the output file
    enum OutputDataType{
        size_field,   // size 
        frame_field,  // vector pair
        metric_field, // dim x dim matrix 
        };
    OutputDataType output_data_type;

    // need to add: isotropy indicators AND smoothness indicator

    // double p; // polynomial order when fixed, should take this from the grid
    double norm_Lq; // for the Lq norm

    double refinement_fraction; // refinement fraction
    double coarsening_fraction; // coarsening fraction

    // new_complexity = (old_complexity * complexity_scale) + complexity_add
    double complexity_scale; // multiplier to complexity
    double complexity_add; // additive to complexity

    std::vector<double> complexity_vector; // vector of complexities

    GridRefinementParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_GRID_REFINEMENT_H__
