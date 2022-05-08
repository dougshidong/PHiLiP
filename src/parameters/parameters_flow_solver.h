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
        };
    FlowCaseType flow_case_type; ///< Selected FlowCaseType from the input file

    double final_time; ///< Final solution time
    double courant_friedrich_lewy_number; ///< Courant-Friedrich-Lewy (CFL) number for constant time step

    /** Name of the output file for writing the unsteady data;
     *  will be written to file: unsteady_data_table_filename.txt */
    std::string unsteady_data_table_filename;

    bool steady_state; ///<Flag for solving steady state solution

    /** Name of the output file for writing the sensitivity data;
     *   will be written to file: sensitivity_table_filename.txt */
    std::string sensitivity_table_filename;

    /** Name of the Gmsh file to be read if the flow_solver_case indeed reads a mesh;
     *  will read file: input_mesh_filename.msh */
    std::string input_mesh_filename;

    /** For taylor green vortex integration tests, expected kinetic energy at final time. */
    double expected_kinetic_energy_at_final_time;

    bool restart_computation_from_file; ///< Restart computation from restart file
    bool output_restart_files; ///< Output the restart files
    std::string restart_files_directory_name; ///< Name of directory for writing and reading restart files
    int restart_file_index; ///< Index of desired restart file for restarting the computation from
    int output_restart_files_every_x_steps; ///< Outputs the restart files every x steps
    double output_restart_files_every_dt_time_intervals; ///< Outputs the restart files at time intervals of dt

    /// For taylor green vortex, selects the type of density initialization
    enum DensityInitialConditionType{
        uniform,
        isothermal,
        };
    /// Selected DensityInitialConditionType from the input file
    DensityInitialConditionType density_initial_condition_type;

    /// For taylor green vortex, flag for computing the integrated enstrophy
    bool output_integrated_enstrophy;

    /// For taylor green vortex, flag for computing the vorticity based dissipation rate
    bool output_vorticity_based_dissipation_rate;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif

