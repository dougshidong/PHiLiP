#ifndef __PARAMETERS_MANUFACTURED_SOLUTION_H__
#define __PARAMETERS_MANUFACTURED_SOLUTION_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameters related to the manufactured convergence study
class ManufacturedSolutionParam
{
public:
    ManufacturedSolutionParam(); ///< Constructor

    /// Uses non-zero source term based on the manufactured solution and the PDE.
    bool use_manufactured_source_term;
    
    /// Selects the manufactured solution to be used if use_manufactured_source_term=true
    enum ManufacturedSolutionType{
        sine_solution,
        cosine_solution,
        additive_solution,
        exp_solution,
        poly_solution,
        even_poly_solution,
        atan_solution,
        boundary_layer_solution,
        s_shock_solution,
        };
    ManufacturedSolutionType manufactured_solution_type; ///< Selected ManufacturedSolutionType from the input file

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_MANUFACTURED_SOLUTION_H__

