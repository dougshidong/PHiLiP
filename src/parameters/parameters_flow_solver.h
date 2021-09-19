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
        inviscid_taylor_green_vortex,
        viscous_taylor_green_vortex
        };
    FlowCaseType flow_case_type; ///< Selected FlowCaseType from the input file
};

} // Parameters namespace

} // PHiLiP namespace

#endif

