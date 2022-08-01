#ifndef __PARAMETERS_FLOW_SOLVER_H__
#define __PARAMETERS_FLOW_SOLVER_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameters related to the flow solver
class FlowSolverParam
{
public:
    FlowSolverParam(); ///< Constructor

    /// Selects the flow case to be simulated
    enum FlowCaseType{
        taylor_green_vortex,
        burgers_viscous_snapshot,
        naca0012,
        burgers_rewienski_snapshot,
        periodic_1D_unsteady,
        gaussian_bump
        };
    FlowCaseType flow_case_type; ///< Selected FlowCaseType from the input file

    unsigned int poly_degree; ///< Polynomial order (P) of the basis functions for DG.
    double final_time; ///< Final solution time
    double courant_friedrich_lewy_number; ///< Courant-Friedrich-Lewy (CFL) number for constant time step

    /** Name of the output file for writing the unsteady data;
     *  will be written to file: unsteady_data_table_filename.txt */
    std::string unsteady_data_table_filename;

    bool steady_state; ///<Flag for solving steady state solution
    bool steady_state_polynomial_ramping; ///< Flag for steady state polynomial ramping

    bool adaptive_time_step; ///< Flag for computing the time step on the fly

    /** Name of the output file for writing the sensitivity data;
     *   will be written to file: sensitivity_table_filename.txt */
    std::string sensitivity_table_filename;

    /** Name of the Gmsh file to be read if the flow_solver_case indeed reads a mesh;
     *  will read file: input_mesh_filename.msh */
    std::string input_mesh_filename;

    bool restart_computation_from_file; ///< Restart computation from restart file
    bool output_restart_files; ///< Output the restart files
    std::string restart_files_directory_name; ///< Name of directory for writing and reading restart files
    int restart_file_index; ///< Index of desired restart file for restarting the computation from
    int output_restart_files_every_x_steps; ///< Outputs the restart files every x steps
    double output_restart_files_every_dt_time_intervals; ///< Outputs the restart files at time intervals of dt

    /// Parameters related to mesh generation
    unsigned int grid_degree; ///< Polynomial degree of the grid
    double grid_left_bound; ///< Left bound of domain for hyper_cube mesh based cases
    double grid_right_bound; ///< Right bound of domain for hyper_cube mesh based cases
    unsigned int number_of_grid_elements_per_dimension; ///< Number of grid elements per dimension for hyper_cube mesh based cases
    int number_of_mesh_refinements; ///< Number of refinements for naca0012 and Gaussian bump based cases

    double channel_height; ///< Height of channel for gaussian bump case
    double channel_length; ///< Width of channel for gaussian bump case
    double bump_height; ///< Height of gaussian bump
    int number_of_subdivisions_in_x_direction; ///< Number of subdivisions in x direction for gaussian bump case
    int number_of_subdivisions_in_y_direction; ///< Number of subdivisions in y direction for gaussian bump case
    int number_of_subdivisions_in_z_direction; ///< Number of subdivisions in z direction for gaussian bump case

    /** For taylor green vortex integration tests, expected kinetic energy at final time. */
    double expected_kinetic_energy_at_final_time;

    /** For taylor green vortex integration tests,
     *  expected theoretical kinetic energy dissipation
     *  rate at final time. */
    double expected_theoretical_dissipation_rate_at_final_time;

    /// For taylor green vortex, selects the type of density initialization
    enum DensityInitialConditionType{
        uniform,
        isothermal,
        };
    /// Selected DensityInitialConditionType from the input file
    DensityInitialConditionType density_initial_condition_type;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif

