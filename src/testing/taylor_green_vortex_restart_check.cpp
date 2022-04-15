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

    dealii::LinearAlgebra::distributed::Vector<double> old_solution(flow_solver->dg->solution);
    old_solution.update_ghost_values();
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(flow_solver->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    flow_solver->dg->triangulation->save("saved-solution");

    // loading the file
    flow_solver->dg->triangulation->load("saved-solution");
    // --- after allocate_dg
    // TO DO: Read section "Note on usage with DoFHandler with hp-capabilities" and add the stuff im missing
    // ------ Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer2(flow_solver->dg->dof_handler);
    flow_solver->dg->solution.zero_out_ghosts();
    solution_transfer2.deserialize(flow_solver->dg->solution);
    flow_solver->dg->solution.update_ghost_values();

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