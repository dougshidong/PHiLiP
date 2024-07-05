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
    else if((this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::turbulent) || 
            (this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured))
    {
        // Turbulent velocity profile using Reichart's law of the wall
        // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
        double dist_from_wall = this->half_channel_height; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [-half_channel_height, half_channel_height]
        if(y > 0.0){
            dist_from_wall -= y; // distance from top wall
        } else if(y < 0.0) {
            dist_from_wall += y; // distance from bottom wall
        } // note: dist_from_wall is non-dimensional

        // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
        // References: 
        /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
            "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

            (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
            on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
        */
        const double kappa = 0.38; // von Karman's constant
        const double C = 4.1;
        
        // STEP 1
        const double reynolds_number_inf = this->all_parameters->navier_stokes_param.reynolds_number_inf;
        const double density = 1.0; // non-dimensional
        const double viscosity_coefficient = this->all_parameters->navier_stokes_param.nondimensionalized_constant_viscosity; // non-dimensional
        const double reynolds_number_based_on_friction_velocity = this->all_parameters->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number;
        const double friction_velocity = reynolds_number_based_on_friction_velocity/reynolds_number_inf; // non-dimensional
        const double y_plus = reynolds_number_inf*density*friction_velocity*dist_from_wall/viscosity_coefficient; // dimensional
        const double u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0)); // dimensional
        const double x_velocity = u_plus*friction_velocity; // non-dimensional
    }
    return x_velocity;
}


template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_x_velocity_gradient(const double y) const 
{
    double x_velocity_gradient = 0.0;
    if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar)
    {
        // Turbulent velocity profile using Reichart's law of the wall
        // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
        double dist_from_wall = this->half_channel_height; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [-half_channel_height, half_channel_height]
        if(y > 0.0){
            dist_from_wall -= y; // distance from top wall
        } else if(y < 0.0) {
            dist_from_wall += y; // distance from bottom wall
        } // note: dist_from_wall is non-dimensional
        x_velocity_gradient = (15.0/2.0)*y*(y*y - this->half_channel_height*this->half_channel_height)/pow(this->half_channel_height,4.0);
    }
    // else if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured)
    // {
    //     x_velocity_gradient = (15.0/2.0)*y*y*y/pow(this->half_channel_height,4.0);
    // }
    else if((this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::turbulent) || 
            (this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured))
    {
        // Turbulent velocity profile using Reichart's law of the wall
        // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
        double dist_from_wall = this->half_channel_height; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [-half_channel_height, half_channel_height]
        if(y > 0.0){
            dist_from_wall -= y; // distance from top wall
        } else if(y < 0.0) {
            dist_from_wall += y; // distance from bottom wall
        } // note: dist_from_wall is non-dimensional

        // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
        // References: 
        /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
            "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

            (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
            on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
        */
        const double kappa = 0.38; // von Karman's constant
        const double C = 4.1;
        
        // STEP 1
        const double reynolds_number_inf = this->all_parameters->navier_stokes_param.reynolds_number_inf;
        const double density = 1.0; // non-dimensional
        const double viscosity_coefficient = this->all_parameters->navier_stokes_param.nondimensionalized_constant_viscosity; // non-dimensional
        const double reynolds_number_based_on_friction_velocity = this->all_parameters->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number;
        const double friction_velocity = reynolds_number_based_on_friction_velocity/reynolds_number_inf; // non-dimensional
        const double y_plus = reynolds_number_inf*density*friction_velocity*dist_from_wall/viscosity_coefficient; // dimensional
        // dimensional
        const double duplus_dyplus = (1.0/kappa)*(kappa/(1.0+kappa*y_plus)) + (C - (1.0/kappa)*log(kappa))*((1.0/11.0)*exp(-y_plus/11.0)+((y_plus-3.0)/33.0)*exp(-y_plus/3.0));

        // STEP 2
        const double du_duplus = friction_velocity; // pulled out non-dim factor for step 3
        const double dyplus_dy = reynolds_number_inf*friction_velocity*density/viscosity_coefficient; // pulled out non-dim factor for step 3

        // STEP 3
        x_velocity_gradient = du_duplus*duplus_dyplus*dyplus_dy; // non-dimensional
    }
    return x_velocity_gradient;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_wall_shear_stress() const
{
    // for constant viscosity we can write:
    const double nondimensionalized_constant_viscosity = this->all_parameters->navier_stokes_param.nondimensionalized_constant_viscosity;
    const double scaled_nondim_viscosity = nondimensionalized_constant_viscosity/this->all_parameters->navier_stokes_param.reynolds_number_inf;
    const double wall_shear_stress_top_wall = scaled_nondim_viscosity*get_x_velocity_gradient(this->y_top_wall)*this->normal_vector_bottom_wall;//this->normal_vector_top_wall;
    const double wall_shear_stress_bottom_wall = scaled_nondim_viscosity*get_x_velocity_gradient(this->y_bottom_wall)*this->normal_vector_bottom_wall;
    const double average_wall_shear_stress = 0.5*(wall_shear_stress_top_wall + wall_shear_stress_bottom_wall);
    return average_wall_shear_stress;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_integral_of_x_velocity(const double y_plus) const
{
    double value = 0.0;
    if((this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::turbulent) || 
            (this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured))
    {
        // Turbulent velocity profile using Reichart's law of the wall

        // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
        // References: 
        /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
            "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

            (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
            on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
        */
        const double kappa = 0.38; // von Karman's constant
        const double C = 4.1;
        
        // Analytical integral expression
        // dimensional
        value = ((1.0/kappa + y_plus)*log(1.0+kappa*y_plus) - y_plus)/kappa;
        value += (C - (1.0/kappa)*log(kappa))*(y_plus + 11.0*exp(-y_plus/11.0) + (3.0/11.0)*(y_plus+3.0)*exp(-y_plus/3.0));
    }
    return value;
}

template <int dim, int nstate>
double TurbulentChannelFlowSkinFrictionCheck<dim, nstate>::get_bulk_velocity() const
{
    double bulk_velocity = 0.0;
    if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::laminar)
    {
        bulk_velocity = 1.0;
    }
    // else if(this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured)
    // {
    //     bulk_velocity = (3.0/8.0);
    // }
    else if((this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::turbulent) || 
            (this->xvelocity_initial_condition_type == XVelocityInitialConditionEnum::manufactured))
    {
        // Turbulent velocity profile using Reichart's law of the wall
        const double reynolds_number_inf = this->all_parameters->navier_stokes_param.reynolds_number_inf;
        const double density = 1.0; // non-dimensional
        const double viscosity_coefficient = this->all_parameters->navier_stokes_param.nondimensionalized_constant_viscosity; // non-dimensional
        const double reynolds_number_based_on_friction_velocity = this->all_parameters->flow_solver_param.turbulent_channel_friction_velocity_reynolds_number;
        const double friction_velocity = reynolds_number_based_on_friction_velocity/reynolds_number_inf; // non-dimensional
        const double y_plus_max = reynolds_number_inf*density*friction_velocity*1.0/viscosity_coefficient; // dimensional
        const double y_plus_min = 0.0; // dimensional
        const double integrand_wrt_y = 2.0*(get_integral_of_x_velocity(y_plus_max) - get_integral_of_x_velocity(y_plus_min)); // dimensional; symmetry applied
        
        // domain
        const double domain_length_x = this->all_parameters->flow_solver_param.turbulent_channel_domain_length_x_direction; // non-dimensional
        const double domain_length_y = this->all_parameters->flow_solver_param.turbulent_channel_domain_length_y_direction; // non-dimensional
        const double domain_length_z = this->all_parameters->flow_solver_param.turbulent_channel_domain_length_z_direction; // non-dimensional
        const double domain_volume = domain_length_x*domain_length_y*domain_length_z; // non-dimensional

        const double volume_integral = integrand_wrt_y*domain_length_x*domain_length_z;

        bulk_velocity = volume_integral/domain_volume;
        // note ref_length = 1.0 anyways so no need to worry about the dim integrand_wrt_y
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
	pcout << "Error is : " << relative_error_skin_friction_coefficient << std::endl;
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
