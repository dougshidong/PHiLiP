#include "taylor_green_vortex_energy_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TaylorGreenVortexEnergyCheck<dim, nstate>::TaylorGreenVortexEnergyCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
        , theoretical_dissipation_rate_expected(parameters_input->flow_solver_param.expected_theoretical_dissipation_rate_at_final_time)
{}

template <int dim, int nstate>
int TaylorGreenVortexEnergyCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // Compute kinetic energy and theoretical dissipation rate
    std::unique_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicTurbulence<dim,nstate>>(this->all_parameters);
    flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver->dg));
    const double kinetic_energy_computed = flow_solver_case->get_integrated_kinetic_energy();
    const double theoretical_dissipation_rate_computed = flow_solver_case->get_vorticity_based_dissipation_rate();

    const double relative_error_kinetic_energy = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    const double relative_error_theoretical_dissipation_rate = abs(theoretical_dissipation_rate_computed - theoretical_dissipation_rate_expected)/theoretical_dissipation_rate_expected;
    if (relative_error_kinetic_energy > 1.0e-10) {
        pcout << "Computed kinetic energy is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    if (relative_error_theoretical_dissipation_rate > 1.0e-10) {
        pcout << "Computed theoretical dissipation rate is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed kinetic energy and theoretical dissipation rate are within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TaylorGreenVortexEnergyCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace