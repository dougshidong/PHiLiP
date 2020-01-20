#ifndef __PARAMETERS_GRID_REFINEMENT_STUDY_H__
#define __PARAMETERS_GRID_REFINEMENT_STUDY_H__

#include <array>

#include <deal.II/base/parameter_handler.h>

#include "parameters/parameters_functional.h"
#include "parameters/parameters_grid_refinement.h"
#include "parameters/parameters_manufactured_solution.h"
#include "parameters/parameters_manufactured_convergence_study.h"
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

class GridRefinementStudyParam
{
    // max number of different refinement procedures
    static const unsigned int MAX_REFINEMENTS = 10;

public:
    GridRefinementStudyParam(); ///< Constructor

    FunctionalParam functional_param;

    ManufacturedSolutionParam manufactured_solution_param;

    // array for refinement steps
    std::array<GridRefinementParam, MAX_REFINEMENTS> grid_refinement_param_vector;

    unsigned int poly_degree;
    unsigned int poly_degree_max;
    unsigned int poly_degree_grid;

    // number of different refinement procedures stored, 0 indicates to use the default pathway
    unsigned int num_refinements; 

    // simplified set of descriptors for the grid for now, replace by grin in param
    using GridEnum = Parameters::ManufacturedConvergenceStudyParam::GridEnum;
    GridEnum grid_type;
    std::string input_grid;
    
    double grid_left; 
    double grid_right;

    unsigned int grid_size;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_GRID_REFINEMENT_STUDY_H__
