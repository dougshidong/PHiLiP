#ifndef __PARAMETERS_MANUFACTURED_CONVERGENCE_STUDY_H__
#define __PARAMETERS_MANUFACTURED_CONVERGENCE_STUDY_H__

#include <deal.II/base/parameter_handler.h>

#include "parameters/parameters_manufactured_solution.h"
#include "parameters/parameters.h"

namespace PHiLiP {
namespace Parameters {

/// Parameters related to the manufactured convergence study
class ManufacturedConvergenceStudyParam
{
public:
    ManufacturedConvergenceStudyParam (); ///< Constructor

    /// Associated manufactured solution parameters
    ManufacturedSolutionParam manufactured_solution_param;

    /// Types of grids that can be used for convergence study.
    /** Hypercube is simply a square from 0,1 in multiple dimensions.
     *  Sinehypercube will take that hypercube and transform it into a sinoisidal pattern
     *  read_grid takes in a set of grids from Gmsh
     */
    enum GridEnum { hypercube, sinehypercube, read_grid };

    /// Grid type
    GridEnum grid_type;

    /// Name of the input grids if read_grid
    /** input_grids-XX.msh, where XX represents the grid number
     */
    std::string input_grids;

    /// Will randomly distort mesh except on boundaries.
    /** Useful sanity check to make sure some errors don't cancel out.
     */
    double random_distortion;
    /// Output the meshes as in a Gmsh format
    bool output_meshes;

    unsigned int degree_start; ///< First polynomial degree to start the loop. If diffusion, must be at least 1.
    unsigned int degree_end; ///< Last polynomial degree to loop.
    unsigned int initial_grid_size; ///< Initial grid size.
    /// Number of grid in the grid study.
    /** Note that for p=0, it will use number_of_grids+2 because for p=0, we need more grids to observe the p+1 convergence
    */
    unsigned int number_of_grids;
    /// Multiplies the last grid size by this amount.
    /** Note that this is the grid progression in 1 dimension.
     *  ith-grid will be of size (initial_grid*(i*grid_progression)+(i*grid_progression_add))^dim")
     */
    double grid_progression;

    /// Adds number of cells to 1D grid.
    /** Note that this is the grid progression in 1 dimension.
     *  ith-grid will be of size (initial_grid*(i*grid_progression)+(i*grid_progression_add))^dim")
     */
    int grid_progression_add;

    /// Tolerance within which the convergence orders are considered to be optimal.
    double slope_deficit_tolerance;

    /// Output the convergence tables (for each p) as txt files; currently only works for tests using grid_study.cpp
    bool output_convergence_tables;

    /// Output the solution files (for each p and grid) as vtu and pvtu files; currently only works for tests using grid_study.cpp
    bool output_solution;

    /// Adds the statewise solution L2 error to the convergence tables; currently only works for tests using grid_study.cpp
    bool add_statewise_solution_error_to_convergence_tables;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif

