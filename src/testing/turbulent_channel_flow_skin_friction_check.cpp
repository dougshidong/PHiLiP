#include "turbulent_channel_flow_skin_friction_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/channel_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::TurbulentChannelFlowSkinFrictionCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , half_channel_height(parameters_input->flow_solver_param.turbulent_channel_domain_length_y_direction/2.0)
{}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_x_velocity(const double y) const 
{
    const double x_velocity = (15.0/8.0)*pow(1.0-pow(y/this->half_channel_height,2.0),2.0);
    return x_velocity;
}


template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_x_velocity_gradient(const double y) const 
{
    const double x_velocity_gradient_wrt_y = (15.0/2.0)*y*(y*y - this->half_channel_height*this->half_channel_height)/pow(this->half_channel_height,4.0);
    return x_velocity_gradient_wrt_y;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::compute_wall_shear_stress() const
{
    // for constant viscosity we can write:
    const double nondimensionalized_constant_viscosity = this->all_parameters.navier_stokes_param.nondimensionalized_constant_viscosity;
    const double scaled_nondim_viscosity = nondimensionalized_constant_viscosity/this->all_parameters.navier_stokes_param.reynolds_number_inf;
    const double wall_shear_stress = scaled_nondim_viscosity*get_x_velocity_gradient(-1.0); // should be the same for both walls
    return wall_shear_stress;
}

template <int dim, int nstate>
int TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // Compute kinetic energy and theoretical dissipation rate
    std::unique_ptr<FlowSolver::ChannelFlow<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::ChannelFlow<dim,nstate>>(this->all_parameters);
    const double wall_shear_stress_computed = flow_solver_case->get_average_wall_shear_stress(*(flow_solver->dg));
    pcout << "computed wall shear stress is " << wall_shear_stress_computed << std::endl;
    pcout << "expected wall shear stress is " << compute_wall_shear_stress() << std::endl;
    // flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver->dg));
    // const double kinetic_energy_computed = flow_solver_case->get_integrated_kinetic_energy();
    // const double theoretical_dissipation_rate_computed = flow_solver_case->get_vorticity_based_dissipation_rate();

    // const double relative_error_kinetic_energy = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    // const double relative_error_theoretical_dissipation_rate = abs(theoretical_dissipation_rate_computed - theoretical_dissipation_rate_expected)/theoretical_dissipation_rate_expected;
    // if (relative_error_kinetic_energy > 1.0e-10) {
    //     pcout << "Computed kinetic energy is not within specified tolerance with respect to expected value." << std::endl;
    //     return 1;
    // }
    // if (relative_error_theoretical_dissipation_rate > 1.0e-10) {
    //     pcout << "Computed theoretical dissipation rate is not within specified tolerance with respect to expected value." << std::endl;
    //     return 1;
    // }
    // pcout << " Test passed, computed kinetic energy and theoretical dissipation rate are within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TurbulentChannelFlowSkinFrictionCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace