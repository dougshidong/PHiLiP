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
        decaying_homogeneous_isotropic_turbulence,
        burgers_viscous_snapshot,
        naca0012,
        burgers_rewienski_snapshot,
        burgers_inviscid,
        convection_diffusion,
        advection,
        periodic_1D_unsteady,
        gaussian_bump,
        sshock
        };
    FlowCaseType flow_case_type; ///< Selected FlowCaseType from the input file

    unsigned int poly_degree; ///< Polynomial order (P) of the basis functions for DG.
    unsigned int max_poly_degree_for_adaptation; ///< Maximum polynomial order of the DG basis functions for adaptation.
    double final_time; ///< Final solution time
    double constant_time_step; ///< Constant time step
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
    unsigned int restart_file_index; ///< Index of desired restart file for restarting the computation from
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
    /// For TGV, flag to calculate and write numerical entropy
    bool do_calculate_numerical_entropy;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

    /// Selects the method for applying the initial condition
    enum ApplyInitialConditionMethod{
        interpolate_initial_condition_function,
        project_initial_condition_function,
        read_values_from_file_and_project
        };
    ApplyInitialConditionMethod apply_initial_condition_method; ///< Selected ApplyInitialConditionMethod from the input file

    /** Filename prefix of the input flow setup file. 
     * Example: 'setup' for files named setup-0000i.dat, where i is the MPI rank. 
     * For initializing the flow with values from a file. 
     * To be set when apply_initial_condition_method is read_values_from_file_and_project. */
    std::string input_flow_setup_filename_prefix;

    bool output_velocity_field_at_fixed_times; ///< Flag for outputting velocity field at fixed times
    std::string output_velocity_field_times_string; ///< String of velocity field output times
    unsigned int number_of_times_to_output_velocity_field; ///< Number of times to output the velocity field
    bool output_vorticity_magnitude_field_in_addition_to_velocity; ///< Flag for outputting vorticity magnitude field in addition to velocity field
    std::string output_flow_field_files_directory_name; ///< Name of directory for writing flow field files
    bool output_solution_files_at_velocity_field_output_times; ///< Flag for outputting solution files at the velocity field output times
};

} // Parameters namespace

} // PHiLiP namespace

#endif

