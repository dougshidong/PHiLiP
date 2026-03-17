#include "turbulent_channel_flow_unsteady_quantity_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/channel_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
TurbulentChannelFlowUnsteadyQuantityCheck<dim, nspecies, nstate>::TurbulentChannelFlowUnsteadyQuantityCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , average_wall_shear_stress_expected(parameters_input->flow_solver_param.expected_average_wall_shear_stress_at_final_time)
        , skin_friction_coefficient_expected(parameters_input->flow_solver_param.expected_skin_friction_coefficient_at_final_time)
        , using_wall_model(parameters_input->using_wall_model)
{}

template <int dim, int nspecies, int nstate>
int TurbulentChannelFlowUnsteadyQuantityCheck<dim, nspecies, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // Compute kinetic energy, enstrophy, and palinstrophy
    std::unique_ptr<FlowSolver::ChannelFlow<dim, nspecies, nstate>> flow_solver_case = std::make_unique<FlowSolver::ChannelFlow<dim,nspecies,nstate>>(this->all_parameters);
    flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver->dg));

    double average_wall_shear_stress = 0.0;
    if(using_wall_model) average_wall_shear_stress = flow_solver_case->get_average_wall_shear_stress_from_wall_model(*(flow_solver->dg));
    else average_wall_shear_stress = flow_solver_case->get_average_wall_shear_stress(*(flow_solver->dg));
    flow_solver_case->set_bulk_flow_quantities(*(flow_solver->dg));
    const double skin_friction_coefficient = flow_solver_case->get_skin_friction_coefficient_from_average_wall_shear_stress(average_wall_shear_stress);

    const double relative_error_average_wall_shear_stress = abs(average_wall_shear_stress - average_wall_shear_stress_expected)/average_wall_shear_stress_expected;
    const double relative_error_skin_friction_coefficient = abs(skin_friction_coefficient - skin_friction_coefficient_expected)/skin_friction_coefficient_expected;

    if (relative_error_average_wall_shear_stress > 1.0e-10) {
        pcout << "Computed average wall shear stress is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    if (relative_error_skin_friction_coefficient > 1.0e-10) {
        pcout << "Computed skin friction coefficient is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed average wall shear stress, and skin friction coefficient are within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TurbulentChannelFlowUnsteadyQuantityCheck<PHILIP_DIM,PHILIP_SPECIES,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace