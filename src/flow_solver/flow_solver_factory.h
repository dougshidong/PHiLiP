#ifndef __FLOW_SOLVER_FACTORY_H__
#define __FLOW_SOLVER_FACTORY_H__

#include "parameters/all_parameters.h"
#include "flow_solver.h"

namespace PHiLiP {
namespace FlowSolver {

/// Create specified flow solver as FlowSolver object 
/** Factory design pattern whose job is to create the correct flow solver
 */
template <int dim, int nstate, int nspecies=1>
class FlowSolverFactory
{
public:
    /// Factory to return the correct flow solver given input file.
    static std::unique_ptr< FlowSolver<dim,nstate> >
        select_flow_case(const Parameters::AllParameters *const parameters_input,
                         const dealii::ParameterHandler &parameter_handler_input);

    /// Recursive factory that will create FlowSolverBase (i.e. FlowSolver<dim,nstate>)
    static std::unique_ptr< FlowSolverBase > 
        create_flow_solver(const Parameters::AllParameters *const parameters_input,
                           const dealii::ParameterHandler &parameter_handler_input);

};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
