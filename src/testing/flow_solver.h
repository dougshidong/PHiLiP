#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
#include "tests.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

// for generate_grid
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

// for the grid:
#include "grid_refinement_study.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver_cases/periodic_cube_flow.h"
#include "flow_solver_cases/periodic_turbulence.h"
#include "flow_solver_cases/1D_burgers_rewienski_snapshot.h"
#include "flow_solver_cases/1d_burgers_viscous_snapshot.h"
#include "flow_solver_cases/naca0012.h"
#include <deal.II/base/table_handler.h>
#include <string>
#include <vector>
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
        using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

/// Selects which flow case to simulate.
template <int dim, int nstate>
class FlowSolver : public TestsBase
{
public:
    /// Constructor.
    FlowSolver(
        const Parameters::AllParameters *const parameters_input, 
        std::shared_ptr<FlowSolverCaseBase<dim, nstate>>,
        const dealii::ParameterHandler &parameter_handler_input);
    
    /// Destructor
    ~FlowSolver() {};

    /// Pointer to Flow Solver Case
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Simply runs the flow solver and returns 0 upon completion
    int run_test () const override;

    /// Initializes the data table from an existing file
    void initialize_data_table_from_file(
        std::string data_table_filename_with_extension,
        const std::shared_ptr <dealii::TableHandler> data_table) const;

    /// Returns the restart filename without extension given a restart index (adds padding appropriately)
    std::string get_restart_filename_without_extension(const int restart_index_input) const;

protected:
    const Parameters::AllParameters all_param; ///< All parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters
    const unsigned int poly_degree; ///< Polynomial order
    const double final_time; ///< Final time of solution

    /// Name of the reference copy of inputted parameters file; for restart purposes
    const std::string input_parameters_file_reference_copy_filename;
    
public:
    /// Pointer to dg so it can be accessed externally.
    std::shared_ptr<DGBase<dim, double>> dg;

    /// Pointer to ode solver so it can be accessed externally.
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

private:
    /** Returns the column names of a dealii::TableHandler object
     *  given the first line of the file */
    std::vector<std::string> get_data_table_column_names(const std::string string_input) const;

    /// Writes a parameter file (.prm) for restarting the computation with
    void write_restart_parameter_file(const int restart_index_input,
                                      const double constant_time_step_input) const;

    /// Converts a double to a string with scientific format and with full precision
    std::string double_to_string(const double value_input) const;

#if PHILIP_DIM>1
    /// Outputs all the necessary restart files
    void output_restart_files(
        const int current_restart_index,
        const double constant_time_step,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const;
#endif
};

/// Create specified flow solver as FlowSolver object 
/** Factory design pattern whose job is to create the correct physics
 */
template <int dim, int nstate>
class FlowSolverFactory
{
public:
    /// Factory to return the correct flow solver given input file.
    static std::unique_ptr< FlowSolver<dim,nstate> >
        create_FlowSolver(const Parameters::AllParameters *const parameters_input,
                          const dealii::ParameterHandler &parameter_handler_input);
};

} // Tests namespace
} // PHiLiP namespace
#endif
