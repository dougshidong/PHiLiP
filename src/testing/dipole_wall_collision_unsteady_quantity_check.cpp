#include "dipole_wall_collision_unsteady_quantity_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/dipole_wall_collision.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
DipoleWallCollisionUnsteadyQuantityCheck<dim, nspecies, nstate>::DipoleWallCollisionUnsteadyQuantityCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
        , enstrophy_expected(parameters_input->flow_solver_param.expected_enstrophy_at_final_time)
        , palinstrophy_expected(parameters_input->flow_solver_param.expected_palinstrophy_at_final_time)
{}

template <int dim, int nspecies, int nstate>
int DipoleWallCollisionUnsteadyQuantityCheck<dim, nspecies, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // Compute kinetic energy, enstrophy, and palinstrophy
    std::unique_ptr<FlowSolver::DipoleWallCollision<dim, nspecies, nstate>> flow_solver_case = std::make_unique<FlowSolver::DipoleWallCollision<dim,nspecies,nstate>>(this->all_parameters);
    flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver->dg));
    const double kinetic_energy_computed = flow_solver_case->get_integrated_incompressible_kinetic_energy();
    const double enstrophy_computed = flow_solver_case->get_integrated_incompressible_enstrophy();
    const double palinstrophy_computed = flow_solver_case->get_integrated_incompressible_palinstrophy();

    const double relative_error_kinetic_energy = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    const double relative_error_enstrophy = abs(enstrophy_computed - enstrophy_expected)/enstrophy_expected;
    const double relative_error_palinstrophy = abs(palinstrophy_computed - palinstrophy_expected)/palinstrophy_expected;
    if (relative_error_kinetic_energy > 1.0e-10) {
        pcout << "Computed kinetic energy is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    if (relative_error_enstrophy > 1.0e-10) {
        pcout << "Computed enstrophy is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    if (relative_error_palinstrophy > 1.0e-10) {
        pcout << "Computed palinstrophy is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed kinetic energy, enstrophy, and palinstrophy are within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==2
    template class DipoleWallCollisionUnsteadyQuantityCheck<PHILIP_DIM,PHILIP_SPECIES,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace