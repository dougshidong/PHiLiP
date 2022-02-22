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

/// Parameters related to collection of grid refinement runs
class GridRefinementStudyParam
{
    /// Maximum number of different refinement procedures
    static const unsigned int MAX_REFINEMENTS = 10;

public:
    GridRefinementStudyParam(); ///< Constructor

    /// Functional parameters to be used with grid refinement study
    FunctionalParam functional_param;

    /// Manufactured solution parameterse to be used with grid refinement study
    ManufacturedSolutionParam manufactured_solution_param;

    /// Array of grid refinement parameters to be run as part of grid refinement study
    std::array<GridRefinementParam, MAX_REFINEMENTS> grid_refinement_param_vector;

    /// Initial solution polynomial degree
    unsigned int poly_degree;
    /// Maximimum allocated solution polynomial degree
    /** Note: Additional head-room above current polynomial order may be needed
      *       for fine-grid adjoint approximation and reconstruction techniques.
      */
    unsigned int poly_degree_max;
    /// Initial grid polynomial degree
    unsigned int poly_degree_grid;

    /// Number of different refinement procedures stored, 0 indicates to use the default pathway
    unsigned int num_refinements; 

    /// simplified set of descriptors for the grid for now, replace by grid in param
    using GridEnum = Parameters::ManufacturedConvergenceStudyParam::GridEnum;
    /// Grid type selection
    GridEnum grid_type;
    /// Input pathway for GridEnum::read_grid type
    std::string input_grid;
    
    /// Lower coordinate bound for GridEnum::hypercube type
    double grid_left; 
    /// Upper coordinate bound for GridEnum::hypercube type
    double grid_right;

    /// Number of initial elements in each axis for GridEnum::hypercube type
    unsigned int grid_size;

    /// Flag to enable interpolation operation
    /** Skips solution step between grid refinements for faster testing of grid refinement methods.
      */
    bool use_interpolation;

    /// Flag to enable approximation of the functional value on a fine grid before refinement run
    bool approximate_functional;
    /// Specified exact functional value for comparison of error convergence
    double functional_value;

    /// Flag to enable output of grid refinement .vtk file
    bool output_vtk;
    /// Flag to enable output of adjoint .vtk file
    bool output_adjoint_vtk;

    /// Flag to enable output of grid refinement solution error convergence
    bool output_solution_error;
    /// Flag to enable output of grid refinement functional error convergence
    bool output_functional_error;

    /// Flag to enable output of gnuplot graph of solution error convergence
    bool output_gnuplot_solution;
    /// Flag to enable output of gnuplot graph of functional error convergence
    bool output_gnuplot_functional;
    /// Flag to enable gnuplot refresh between iteration runs
    bool refresh_gnuplot;

    /// Flag to enable output of grid refinement wall-clock solution time
    bool output_solution_time;
    /// Flag to enable output of grid refinement wall-clock adjoint time
    bool output_adjoint_time;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_GRID_REFINEMENT_STUDY_H__
