#ifndef __FLOW_SOLVER_CHANNEL_FLOW_H__
#define __FLOW_SOLVER_CHANNEL_FLOW_H__

#include "flow_solver.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class FlowSolverChannelFlow : public FlowSolver<dim,nstate>
{
public:
    /// Constructor.
    FlowSolverChannelFlow(
        const Parameters::AllParameters *const parameters_input, 
        std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~FlowSolverChannelFlow() {};

protected:
    /// Update model variables
    void update_model_variables() override;

private:
    /// Get the integrated density over the domain
    double get_integrated_density_over_domain(DGBase<dim, double> &dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
