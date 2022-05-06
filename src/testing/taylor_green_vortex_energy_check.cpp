#include "taylor_green_vortex_energy_check.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_cube_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TaylorGreenVortexEnergyCheck<dim, nstate>::TaylorGreenVortexEnergyCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template <int dim, int nstate>
int TaylorGreenVortexEnergyCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run_test());

    // Compute kinetic energy
    std::unique_ptr<PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<PeriodicTurbulence<dim,nstate>>(this->all_parameters);
    const double kinetic_energy_computed = flow_solver_case->compute_kinetic_energy(*(flow_solver->dg));

    const double relative_error = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    if (relative_error > 1.0e-10) {
        pcout << "Computed kinetic energy is not within specified tolerance with respect to expected kinetic energy." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed kinetic energy is within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TaylorGreenVortexEnergyCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace