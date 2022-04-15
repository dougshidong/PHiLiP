#include "taylor_green_vortex_restart_check.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_cube_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TaylorGreenVortexRestartCheck<dim, nstate>::TaylorGreenVortexRestartCheck(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template <int dim, int nstate>
int TaylorGreenVortexRestartCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(this->all_parameters);
    static_cast<void>(flow_solver->run_test());

    // Compute kinetic energy
    std::unique_ptr<PeriodicCubeFlow<dim, nstate>> flow_solver_case = std::make_unique<PeriodicCubeFlow<dim,nstate>>(this->all_parameters);
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
    template class TaylorGreenVortexRestartCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace