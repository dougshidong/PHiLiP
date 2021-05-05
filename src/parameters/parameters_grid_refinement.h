#ifndef __PARAMETERS_GRID_REFINEMENT_H__
#define __PARAMETERS_GRID_REFINEMENT_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameters related to individual grid refinement run
class GridRefinementParam
{
public:
    /// Number of refinement steps to be performed
    unsigned int refinement_steps;

    // main set of parameters for deciding the method

    /// Controls the underlying method of refinement
    enum RefinementMethod{
        uniform,        // all cells are refined
        fixed_fraction, // picks fraction with largest indicators
        continuous,     // generates a new mesh based on a size field
        };
    /// Selected method of refinement
    RefinementMethod refinement_method;

    /// Controls the type of refinement to be performed
    enum RefinementType{
        h,  // element size only
        p,  // polynomial orders
        hp, // mix of both
        };
    /// Selected type of refinement to be performed
    RefinementType refinement_type;
    
    /// Flag for performing anisotropic refinement
    /** Note: only availible for some fixed-fraction and continuous method cases.
      *       Also not availible for certain mesh types.
      */
    bool anisotropic;

    /// Maximum anisotropic ratio for continuous size field targets
    double anisotropic_ratio_max;

    /// Minimum anisotropic ratio for continuous zie field targets
    double anisotropic_ratio_min;

    /// threshold value in anisotropic indicator to enable anisotropic splitting 
    /** Note: used only for fixed-fraction anisotropic splitting methods if allowed by mesh type.
      */
    double anisotropic_threshold_ratio;

    /// Control of anisotropic splitting indicator to be used in fixed-fraction methods
    enum AnisoIndicator{
        jump_based,           // based on face jumps with neighbouring cells
        reconstruction_based, // based on p+1 derivative along tensor product chord lines
        };
    /// Selected anisotropic splitting indicator
    AnisoIndicator anisotropic_indicator;

    /// Types of error indicator to be used in the grid refinement
    enum ErrorIndicator{
        error_based,    // using the exact error for testing
        hessian_based,  // feature_based
        residual_based, // requires a fine grid projection
        adjoint_based,  // adjoint based
        };
    /// Selected error indicator type
    ErrorIndicator error_indicator;    

    /// File type/interface to be used for access to external tools
    enum OutputType{
        gmsh_out, // output of pos and geo files for gmsh remeshing
        msh_out,  // output of .msh with data fields corresponding to output_data_type
        };
    /// Selected file output type
    OutputType output_type;

    /// Method of data storage in the output file for continuous methods
    enum OutputDataType{
        size_field,   // size 
        frame_field,  // vector pair
        metric_field, // dim x dim matrix 
        };
    /// Selected data storage type
    OutputDataType output_data_type;

    // need to add: isotropy indicators AND smoothness indicator

    // double p; // polynomial order when fixed, should take this from the grid
    double norm_Lq; ///< Lq norm exponent selection

    double refinement_fraction; ///< refinement fraction for fixed-fraction methods
    double coarsening_fraction; ///< coarsening fraction for fixed-fraction methods

    double r_max; ///< refinement factor for log DWR size field
    double c_max; ///< coarsening factor for log DWR size field

    // new_complexity = (old_complexity * complexity_scale) + complexity_add
    double complexity_scale; ///< multiplier to complexity between grid refinement iterations
    double complexity_add;   ///< additive  constant to complexity between grid refinement iterations

    /// Vector of complexities to be used for initial continuous grid refinement iterations
    /** Note: growth will resort to complexity_scale and complexity_add controls if the initial
      *       vector set is exceeded by refinement_steps.
      */
    std::vector<double> complexity_vector; 

    /// Flag to exit after call to refinement
    bool exit_after_refine;

    GridRefinementParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_GRID_REFINEMENT_H__
