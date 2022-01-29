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
        burgers_rewienski_snapshot,
        };
    FlowCaseType flow_case_type; ///< Selected FlowCaseType from the input file

    double final_time; ///< Final solution time
    double courant_friedrich_lewy_number; ///< Courant-Friedrich-Lewy (CFL) number for constant time step

    /** Name of the output file for writing the unsteady data; 
        will be written to file: unsteady_data_table_filename.txt */
    std::string unsteady_data_table_filename;

    bool steady_state; ///<Flag for solving steady state solution

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif

