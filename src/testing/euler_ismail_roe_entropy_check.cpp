#include "euler_ismail_roe_entropy_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerIsmailRoeEntropyCheck<dim, nstate>::EulerIsmailRoeEntropyCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int EulerIsmailRoeEntropyCheck<dim, nstate>::run_test() const
{
    // Initialize flow_solver
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);

    // Compute kinetic energy and theoretical dissipation rate
    std::unique_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicTurbulence<dim,nstate>>(this->all_parameters);
    const double initial_entropy = flow_solver_case->get_numerical_entropy(flow_solver->dg); 
    static_cast<void>(flow_solver->run());
    const double final_entropy = flow_solver_case->get_numerical_entropy(flow_solver->dg); 

    pcout << "Initial num. entropy: " << std::setprecision(16) << initial_entropy 
          << " final: " << final_entropy 
          << " scaled difference " << abs((initial_entropy-final_entropy)/initial_entropy) 
          << std::endl;
    if (abs((initial_entropy-final_entropy)/initial_entropy) > 1E-15){
        pcout << "Entropy change is not within allowable tolerance. Test failing.";
        return 1;
    }

    return 0;
}

#if PHILIP_DIM==3
    template class EulerIsmailRoeEntropyCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
