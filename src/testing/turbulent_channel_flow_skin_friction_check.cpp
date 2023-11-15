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
        , xvelocity_initial_condition_type(parameters_input->flow_solver_param.xvelocity_initial_condition_type)
        , y_top_wall(1.0)
        , y_bottom_wall(-1.0)
        , normal_vector_top_wall(-1.0)
        , normal_vector_bottom_wall(1.0)
{}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_x_velocity(const double y) const 
{
    double x_velocity = 0.0;
    if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar)
    {
        x_velocity = (15.0/8.0)*pow(1.0-pow(y/this->half_channel_height,2.0),2.0);
    }
    else if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured)
    {
        x_velocity = (15.0/8.0)*pow(y/this->half_channel_height,4.0);
    }
    return x_velocity;
}


template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_x_velocity_gradient(const double y) const 
{
    double x_velocity_gradient = 0.0;
    if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar)
    {
        x_velocity_gradient = (15.0/2.0)*y*(y*y - this->half_channel_height*this->half_channel_height)/pow(this->half_channel_height,4.0);
    }
    else if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured)
    {
        x_velocity_gradient = (15.0/2.0)*y*y*y/pow(this->half_channel_height,4.0);
    }
    return x_velocity_gradient;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_wall_shear_stress() const
{
    // for constant viscosity we can write:
    const double nondimensionalized_constant_viscosity = this->all_parameters->navier_stokes_param.nondimensionalized_constant_viscosity;
    const double scaled_nondim_viscosity = nondimensionalized_constant_viscosity/this->all_parameters->navier_stokes_param.reynolds_number_inf;
    const double wall_shear_stress_top_wall = scaled_nondim_viscosity*get_x_velocity_gradient(this->y_top_wall)*this->normal_vector_top_wall;
    const double wall_shear_stress_bottom_wall = scaled_nondim_viscosity*get_x_velocity_gradient(this->y_bottom_wall)*this->normal_vector_bottom_wall;
    const double average_wall_shear_stress = 0.5*(wall_shear_stress_top_wall + wall_shear_stress_bottom_wall);
    return average_wall_shear_stress;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_bulk_velocity() const
{
    double bulk_velocity = 0.0;
    if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar)
    {
        bulk_velocity = 1.0;
    }
    else if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured)
    {
        bulk_velocity = (3.0/8.0);
    }
    
    return bulk_velocity;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_skin_friction_coefficient() const
{
    // Reference: Equation 34 of Lodato G, Castonguay P, Jameson A. Discrete filter operators for large-eddy simulation using high-order spectral difference methods. International Journal for Numerical Methods in Fluids2013;72(2):231–258. 
    const double avg_wall_shear_stress = this->get_wall_shear_stress();
    const double bulk_density = 1.0; // based on initial condition
    const double bulk_velocity = this->get_bulk_velocity();
    const double skin_friction_coefficient = 2.0*avg_wall_shear_stress/(bulk_density*bulk_velocity*bulk_velocity);
    return skin_friction_coefficient;
}

template <int dim, int nstate>
int TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::run_test() const
{
    // Integrate to final time
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    static_cast<void>(flow_solver->run());

    // Compute kinetic energy and theoretical dissipation rate
    std::unique_ptr<FlowSolver::ChannelFlow<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::ChannelFlow<dim,nstate>>(this->all_parameters);
    const double computed_wall_shear_stress = flow_solver_case->get_average_wall_shear_stress(*(flow_solver->dg));
    const double expected_wall_shear_stress = this->get_wall_shear_stress();
    const double relative_error_wall_shear_stress = abs(computed_wall_shear_stress - expected_wall_shear_stress);
    pcout << "computed wall shear stress is " << computed_wall_shear_stress << std::endl; // remove
    pcout << "expected wall shear stress is " << expected_wall_shear_stress << std::endl; // remove
    pcout << "error is " << relative_error_wall_shear_stress << std::endl;
    if (relative_error_wall_shear_stress > 1.0e-9) {
        pcout << "Computed wall shear stress is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    flow_solver_case->set_bulk_flow_quantities(*(flow_solver->dg));
    const double computed_bulk_velocity = flow_solver_case->get_bulk_velocity();
    const double computed_skin_friction_coefficient = flow_solver_case->get_skin_friction_coefficient_from_average_wall_shear_stress(computed_wall_shear_stress);
    const double expected_bulk_velocity = this->get_bulk_velocity();
    const double expected_skin_friction_coefficient = this->get_skin_friction_coefficient();
    const double relative_error_bulk_velocity = abs(computed_bulk_velocity - expected_bulk_velocity);
    if (relative_error_bulk_velocity > 1.0e-9) {
        pcout << "Computed bulk velocity is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    const double relative_error_skin_friction_coefficient = abs(computed_skin_friction_coefficient - expected_skin_friction_coefficient);
    if (relative_error_skin_friction_coefficient > 1.0e-9) {
        pcout << "Computed skin friction coefficient is not within specified tolerance with respect to expected value." << std::endl;
        return 1;
    }
    pcout << " Test passed, computed wall shear stress and skin friction coefficient are within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class TurbulentChannelFlowSkinFrictionCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace