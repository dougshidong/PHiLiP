#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include "euler_vortex_advection_error_study.h"

#include "physics/initial_conditions/initial_condition_function.h" 
// #include "physics/euler.h"

#include "flow_solver/flow_solver_factory.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerVortexAdvectionErrorStudy<dim,nstate>::EulerVortexAdvectionErrorStudy(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int EulerVortexAdvectionErrorStudy<dim,nstate>
::run_test () const
{
    int convergence_order_achieved = 0;
    std::cout << "DO YOU SEE THIS? \n \n " << std::endl;

    double run_test_output = run_error_study(); // run_error_study() can return either enthalpy or the convergence order
    if (abs(run_test_output) > 1e-15) // If run_error_study() returns non zero
    {
        convergence_order_achieved = 1;  // test failed
    }
    return convergence_order_achieved;
}

template <int dim, int nstate>
double EulerVortexAdvectionErrorStudy<dim,nstate>
::compute_pressure ( const std::array<double,nstate> &conservative_soln ) const
{
    double pressure = conservative_soln[0];

    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = param.flow_solver_param.flow_case_type;
    // Euler
    if (flow_type == FlowCaseEnum::euler_vortex_advection || flow_type == FlowCaseEnum::euler_bubble_advection)
    {
        if constexpr (dim==1 && nstate==dim+2)
        {
            // std::cout << "euler_vortex_advection! \n \n " << std::endl;
            Physics::Euler<dim,nstate,double> euler_physics_double
            = Physics::Euler<dim, nstate, double>(
                &param,
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);

            pressure =  euler_physics_double.compute_pressure(conservative_soln);
        }
    }
    // Multi-Species Calorically-Imperfect Euler Vortex
    else if (flow_type == FlowCaseEnum::multi_species_vortex_advection || 
             flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            pressure =  realgas_physics_double.compute_mixture_pressure(conservative_soln);
        }
    }
    // Multi-Species Calorically-Perfect Euler
    else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically_perfect_vortex_advection! \n \n " << std::endl;
            Physics::MultiSpeciesCaloricallyPerfect<dim,nstate,double> multispecies_calorically_perfect_physics_double
            = Physics::MultiSpeciesCaloricallyPerfect<dim, nstate, double>(
                &param);
            pressure =  multispecies_calorically_perfect_physics_double.compute_mixture_pressure(conservative_soln);
        }
    }
    // 2D Multi-Species Calorically-Imperfect Euler Vortex
    else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection) 
    {
        if constexpr (dim==2 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            pressure =  realgas_physics_double.compute_mixture_pressure(conservative_soln);
        }
    }
    

    return pressure;
}

template <int dim, int nstate>
double EulerVortexAdvectionErrorStudy<dim,nstate>
::compute_temperature ( const std::array<double,nstate> &conservative_soln ) const
{
    double temperature = conservative_soln[0];

    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = param.flow_solver_param.flow_case_type;
    // Euler
    if (flow_type == FlowCaseEnum::euler_vortex_advection || flow_type == FlowCaseEnum::euler_bubble_advection)
    {
        if constexpr (dim==1 && nstate==dim+2)
        {
            // std::cout << "euler_vortex_advection! \n \n " << std::endl;
            Physics::Euler<dim,nstate,double> euler_physics_double
            = Physics::Euler<dim, nstate, double>(
                &param,
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);

            const double pressure =  euler_physics_double.compute_pressure(conservative_soln);
            const double density = conservative_soln[0];
            temperature = euler_physics_double.compute_temperature_from_density_pressure(density,pressure);
        }
    }
    // Multi-Species Calorically-Imperfect Euler
    else if (flow_type == FlowCaseEnum::multi_species_vortex_advection ||
             flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            temperature =  realgas_physics_double.compute_temperature(conservative_soln);
        }
    }
    // Multi-Species Calorically-Perfect Euler
    else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically_perfect_vortex_advection! \n \n " << std::endl;
            Physics::MultiSpeciesCaloricallyPerfect<dim,nstate,double> multispecies_calorically_perfect_physics_double
            = Physics::MultiSpeciesCaloricallyPerfect<dim, nstate, double>(
                &param);
            temperature =  multispecies_calorically_perfect_physics_double.compute_temperature(conservative_soln);
        }
    }
    // 2D Multi-Species Calorically-Imperfect Euler
    else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection) 
    {
        if constexpr (dim==2 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            temperature =  realgas_physics_double.compute_temperature(conservative_soln);
        }
    }
    
    return temperature;
}

template <int dim, int nstate>
double EulerVortexAdvectionErrorStudy<dim,nstate>
::compute_mass_fractions_1st ( const std::array<double,nstate> &conservative_soln ) const
{
    double mass_fraction = conservative_soln[0];

    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = param.flow_solver_param.flow_case_type;
    // Euler
    if (flow_type == FlowCaseEnum::euler_vortex_advection || flow_type == FlowCaseEnum::euler_bubble_advection)
    {
        if constexpr (dim==1 && nstate==dim+2)
        {
            // std::cout << "euler_vortex_advection! \n \n " << std::endl;
            Physics::Euler<dim,nstate,double> euler_physics_double
            = Physics::Euler<dim, nstate, double>(
                &param,
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);

            mass_fraction = conservative_soln[0]; // TO DO: it is dummy 
        }
    }
    // Multi-Species Calorically-Imperfect Multi-Species
    else if (flow_type == FlowCaseEnum::multi_species_vortex_advection ||
             flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            mass_fraction = realgas_physics_double.compute_mass_fractions(conservative_soln)[0];
        }
    }
    // Multi-Species Calorically-Perfect Euler
    else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection) 
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically_perfect_vortex_advection! \n \n " << std::endl;
            Physics::MultiSpeciesCaloricallyPerfect<dim,nstate,double> multispecies_calorically_perfect_physics_double
            = Physics::MultiSpeciesCaloricallyPerfect<dim, nstate, double>(
                &param);
            mass_fraction = multispecies_calorically_perfect_physics_double.compute_mass_fractions(conservative_soln)[0];
        }
    }
    // 2D Multi-Species Calorically-Imperfect Multi-Species
    else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection) 
    {
        if constexpr (dim==2 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);
            mass_fraction = realgas_physics_double.compute_mass_fractions(conservative_soln)[0];
        }
    }
    
    return mass_fraction;
}

template <int dim, int nstate>
double EulerVortexAdvectionErrorStudy<dim,nstate>
::compute_exact_at_q ( const dealii::Point<dim,double> &point, const unsigned int istate ) const
{
    double value = 0.00;

    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = param.flow_solver_param.flow_case_type;
    // Euler
    if (flow_type == FlowCaseEnum::euler_bubble_advection)
    {
        if constexpr (dim==1 && nstate==dim+2)
        {
            // std::cout << "euler_vortex_advection! \n \n " << std::endl;
            Physics::Euler<dim,nstate,double> euler_physics_double
            = Physics::Euler<dim, nstate, double>(
                &param,
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);

            const double speed = 25.0;
            const double cycle = 1.0/100000.0;
            const double moved = speed*cycle;

            const double relax = 0.75;
            const double x = point[0] - moved;
            const double xi = 10.0/2.0;
            const double T_0 = 300.0; // [K]
            const double theta = 2.0;

            const double pressure = 100000; // [N/m^2] -> [bar]
            const double velocity = 50.0/2.0; // [m/s]
            const double temperature = 0.5*((1.0+theta) +(1.0-theta)*tanh(relax*(abs(x)-xi)))*T_0; // [K]
            const double density = pressure/(euler_physics_double.R_Air_Dim*temperature);

            std::array<double,nstate> soln_primitive;

            soln_primitive[0] = density / euler_physics_double.density_ref;
            soln_primitive[1] = velocity / euler_physics_double.u_ref;
            soln_primitive[2] = pressure / (euler_physics_double.density_ref*euler_physics_double.u_ref_sqr);

            const std::array<double,nstate> soln_conservative = euler_physics_double.convert_primitive_to_conservative(soln_primitive);
            if(istate==0) {
            // mixture density
            value = soln_conservative[istate];
            }
            if(istate==1) {
            // x-velocity
            value = soln_conservative[istate];
            }
            if(istate==2) {
            // pressure
            value = soln_conservative[istate];
            }
        
        }
    }

    // Multi-Species Calorically-Perfect Euler Vortex
    else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection)
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically_perfect_vortex_advection! \n \n " << std::endl;
            Physics::MultiSpeciesCaloricallyPerfect<dim,nstate,double> multispecies_calorically_perfect_physics_double
            = Physics::MultiSpeciesCaloricallyPerfect<dim, nstate, double>(
                &param);
 
        if constexpr(dim == 1) {
            const double speed = 10.0;
            const double t_end = 8.65e-1;
            const double t_cycle = multispecies_calorically_perfect_physics_double.u_ref*0.10;
            const double cycle = t_end/t_cycle;
            const double moved = speed*cycle;

        const double x = point[0] - moved;
        const double x_0 = 5.0;
        const double r = sqrt((x-x_0)*(x-x_0));
        const double T_0 = 300.0; // [K]
        const double big_gamma = 50.0;
        const double gamma_0 = 1.4;
        const double y_H2_0 = 0.01277;
        // const double y_O2_0 = 0.101;
        const double a_1 = 0.005;
        // const double a_2 = 0.03;
        const double pi = 6.28318530717958623200 / 2; // pi

        const double pressure = 101325; // [N/m^2]
        const double velocity = 100.0; // [m/s]
        const double exp = std::exp(0.50*(1-r*r));
        const double coeff = 2*pi/(gamma_0*big_gamma);
        const double temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const double y_H2 = (y_H2_0 - a_1*coeff*exp);
        const double y_O2 = 1.0 - y_H2;
        // const double y_O2 = (y_O2_0 - a_2*coeff*exp);
        // const double y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = multispecies_calorically_perfect_physics_double.compute_Rs(multispecies_calorically_perfect_physics_double.Ru);
        const double R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*multispecies_calorically_perfect_physics_double.R_ref;
        const double density = pressure/(R_mixture*temperature);

            std::array<double,nstate> soln_primitive;

            soln_primitive[0] = density / multispecies_calorically_perfect_physics_double.density_ref;
            soln_primitive[1] = velocity / multispecies_calorically_perfect_physics_double.u_ref;
            soln_primitive[2] = pressure / (multispecies_calorically_perfect_physics_double.density_ref*multispecies_calorically_perfect_physics_double.u_ref_sqr);
            soln_primitive[3] = y_H2;

            const std::array<double,nstate> soln_conservative = multispecies_calorically_perfect_physics_double.convert_primitive_to_conservative(soln_primitive);
            if(istate==0) {
            // mixture density
            value = soln_conservative[istate];
            }
            if(istate==1) {
            // x-velocity
            value = soln_conservative[istate];
            }
            if(istate==2) {
            // pressure
            value = soln_conservative[istate];
            }
            if(istate==3) {
            // Y_H2
            value = soln_conservative[istate];
            }
    }
        
        }
    }    

    // Multi-Species Euler (Calorically Imperfect) Vortex
    else if (flow_type == FlowCaseEnum::multi_species_vortex_advection)
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);

            const double speed = 10.0;
            const double t_end = 34.614724700315755e-1;
            const double t_cycle = realgas_physics_double.u_ref*0.10;
            const double cycle = t_end/t_cycle;
            const double moved = speed*cycle;

        const double x = point[0] - moved;
        const double x_0 = 5.0;
        const double r = sqrt((x-x_0)*(x-x_0));
        const double T_0 = 300.0; // [K]
        const double big_gamma = 50.0;
        const double gamma_0 = 1.4;
        const double y_H2_0 = 0.01277;
        // const double y_O2_0 = 0.101;
        const double a_1 = 0.005;
        // const double a_2 = 0.03;
        const double pi = 6.28318530717958623200 / 2; // pi

        const double pressure = 101325; // [N/m^2]
        const double velocity = 100.0; // [m/s]
        const double exp = std::exp(0.50*(1-r*r));
        const double coeff = 2*pi/(gamma_0*big_gamma);
        const double temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const double y_H2 = (y_H2_0 - a_1*coeff*exp);
        const double y_O2 = 1.0 - y_H2;
        // const double y_O2 = (y_O2_0 - a_2*coeff*exp);
        // const double y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = realgas_physics_double.compute_Rs(realgas_physics_double.Ru);
        const double R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*realgas_physics_double.R_ref;
        const double density = pressure/(R_mixture*temperature);

            std::array<double,nstate> soln_primitive;

            soln_primitive[0] = density / realgas_physics_double.density_ref;
            soln_primitive[1] = velocity / realgas_physics_double.u_ref;
            soln_primitive[2] = pressure / (realgas_physics_double.density_ref*realgas_physics_double.u_ref_sqr);
            soln_primitive[3] = y_H2;

            const std::array<double,nstate> soln_conservative = realgas_physics_double.convert_primitive_to_conservative(soln_primitive);
            if(istate==0) {
            // mixture density
            value = soln_conservative[istate];
            }
            if(istate==1) {
            // x-velocity
            value = soln_conservative[istate];
            }
            if(istate==2) {
            // pressure
            value = soln_conservative[istate];
            }
            if(istate==3) {
            // Y_H2
            value = soln_conservative[istate];
            }
        
        }
    }    

    // Multi-Species Euler (Calorically Imperfect, High_temperature) Vortex
    else if (flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection)
    {
        if constexpr (dim==1 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);

            const double speed = 10.0;
            const double t_end = 34.614724700315755e-1;
            const double t_cycle = realgas_physics_double.u_ref*0.10;
            const double cycle = t_end/t_cycle;
            const double moved = speed*cycle;

        const double x = point[0] - moved;
        const double x_0 = 5.0;
        const double r = sqrt((x-x_0)*(x-x_0));
        const double T_0 = 300.0; // [K]
        const double big_gamma = 50.0;
        const double gamma_0 = 1.4;
        const double y_H2_0 = 0.01277;
        // const double y_O2_0 = 0.101;
        const double a_1 = 0.005;
        // const double a_2 = 0.03;
        const double pi = 6.28318530717958623200 / 2; // pi

        double pressure = 101325; // [N/m^2]
        pressure *= 5.0;
        const double velocity = 100.0; // [m/s]
        const double exp = std::exp(0.50*(1-r*r));
        const double coeff = 2*pi/(gamma_0*big_gamma);
        double temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        temperature *= 5.0;
        const double y_H2 = (y_H2_0 - a_1*coeff*exp);
        const double y_O2 = 1.0 - y_H2;
        // const double y_O2 = (y_O2_0 - a_2*coeff*exp);
        // const double y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = realgas_physics_double.compute_Rs(realgas_physics_double.Ru);
        const double R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*realgas_physics_double.R_ref;
        const double density = pressure/(R_mixture*temperature);

            std::array<double,nstate> soln_primitive;

            soln_primitive[0] = density / realgas_physics_double.density_ref;
            soln_primitive[1] = velocity / realgas_physics_double.u_ref;
            soln_primitive[2] = pressure / (realgas_physics_double.density_ref*realgas_physics_double.u_ref_sqr);
            soln_primitive[3] = y_H2;

            const std::array<double,nstate> soln_conservative = realgas_physics_double.convert_primitive_to_conservative(soln_primitive);
            if(istate==0) {
            // mixture density
            value = soln_conservative[istate];
            }
            if(istate==1) {
            // x-velocity
            value = soln_conservative[istate];
            }
            if(istate==2) {
            // pressure
            value = soln_conservative[istate];
            }
            if(istate==3) {
            // Y_H2
    // 2D Multi-Species Euler (Calorically Imperfect) Vortex
    else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection)
    {
        if constexpr (dim==2 && nstate==dim+2+2-1) // TO DO: N_SPECIES, dim = 1, nstate = dim+2+3-1
        {
            // std::cout << "multi-species_calorically-imperfect_vortex_advection! \n \n " << std::endl;
            Physics::RealGas<dim,nstate,double> realgas_physics_double
            = Physics::RealGas<dim, nstate, double>(
                &param);

            const double speed = 10.0;
            const double t_end = 34.614724700315755e-2;
            const double t_cycle = realgas_physics_double.u_ref*0.10;
            const double cycle = t_end/t_cycle;
            const double moved = speed*cycle;

        const double x = point[0] - moved;
        const double y = point[1] - moved;
        const double x_0 = 5.0;
        const double y_0 = 5.0;
        const double r = sqrt((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0));
        // const double r = sqrt((x-x_0)*(x-x_0) + (x-x_0)*(x-x_0));
        const double T_0 = 300.0; // [K]
        const double big_gamma = 50.0;
        const double gamma_0 = 1.4;
        const double y_H2_0 = 0.01277;
        // const double y_O2_0 = 0.101;
        const double a_1 = 0.005;
        // const double a_2 = 0.03;
        const double pi = 6.28318530717958623200 / 2; // pi
        const double pressure = 101325; // [N/m^2]
        const double velocity = 100.0; // [m/s]
        const double exp = std::exp(0.50*(1-r*r));
        const double coeff = 2*pi/(gamma_0*big_gamma);
        const double temperature = T_0 - (gamma_0-1.0)*big_gamma*big_gamma/(8.0*gamma_0*pi)*exp;
        const double y_H2 = (y_H2_0 - a_1*coeff*exp);
        const double y_O2 = 1.0 - y_H2;
        // const double y_O2 = (y_O2_0 - a_2*coeff*exp);
        // const double y_N2 = 1.0 - y_H2 - y_O2;

        const std::array Rs = realgas_physics_double.compute_Rs(realgas_physics_double.Ru);
        const double R_mixture = (y_H2*Rs[0] + y_O2*Rs[1])*realgas_physics_double.R_ref;
        const double density = pressure/(R_mixture*temperature);

            std::array<double,nstate> soln_primitive;

            soln_primitive[0] = density / realgas_physics_double.density_ref;
            soln_primitive[1] = velocity / realgas_physics_double.u_ref;
            soln_primitive[2] = velocity / realgas_physics_double.u_ref;
            soln_primitive[3] = pressure / (realgas_physics_double.density_ref*realgas_physics_double.u_ref_sqr);
            soln_primitive[4] = y_H2;

            const std::array<double,nstate> soln_conservative = realgas_physics_double.convert_primitive_to_conservative(soln_primitive);
            // if(istate==0) {
            // // mixture density
            value = soln_conservative[istate];
            // }
            // if(istate==1) {
            // // x-velocity
            // value = soln_conservative[istate];
            // }
            // if(istate==2) {
            // // y-velocity
            // value = soln_conservative[istate];
            // }
            // if(istate==3) {
            // // pressure
            // value = soln_conservative[istate];
            // }
            // if(istate==4) {
            // // Y_H2
            // value = soln_conservative[istate];
            // }
        
        }
    }   
    
    return value;
}

template<int dim, int nstate>
double EulerVortexAdvectionErrorStudy<dim,nstate>
::run_error_study() const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    // output file 
    // error
    // density
    std::ofstream outdata_density;
    outdata_density.open("density_error.txt");
    //  pressure
    std::ofstream outdata_pressure;
    outdata_pressure.open("pressure_error.txt");
    //  temperature
    std::ofstream outdata_temperature;
    outdata_temperature.open("temperature_error.txt");    
    //  mass_fractions_1st
    std::ofstream outdata_mass_fractions_1st;
    outdata_mass_fractions_1st.open("mass_fractions_1st_error.txt");   
    // slope
    // density
    std::ofstream outdata_slope_density;
    outdata_slope_density.open("slope_density_error.txt");
    //  pressure
    std::ofstream outdata_slope_pressure;
    outdata_slope_pressure.open("slope_pressure_error.txt");
    //  temperature
    std::ofstream outdata_slope_temperature;
    outdata_slope_temperature.open("slope_temperature_error.txt");    
    //  mass_fractions_1st
    std::ofstream outdata_slope_mass_fractions_1st;
    outdata_slope_mass_fractions_1st.open("slope_mass_fractions_1st_error.txt"); 


    // // flowtype
    // using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    // const FlowCaseEnum flow_type = param.flow_solver_param.flow_case_type;
    // if (flow_type == FlowCaseEnum::euler_vortex_advection)
    // {
    //     std::cout << "euler_vortex_advection! \n \n " << std::endl;    
    // } 
    // else if (flow_type == FlowCaseEnum::multi_species_vortex_advection)
    // {
    //     std::cout << "multi-species_vortex_advection! \n \n " << std::endl; 
    // }
    // else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection)
    // {
    //     std::cout << "multi-species_calorically_perfect_vortex_advection! \n \n " << std::endl; 
    // }
    // else if (flow_type == FlowCaseEnum::euler_bubble_advection)
    // {
    //     std::cout << "euler_bubble_advection! \n \n " << std::endl; 
    // }

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    // Assert(dim == 2, dealii::ExcDimensionMismatch(dim, param.dimension)); // NOTE: this was originally on for bump case
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids             = manu_grid_conv_param.number_of_grids;

    // Physics::Euler<dim,nstate,double> euler_physics_double
    //     = Physics::Euler<dim, nstate, double>(
    //             &param,
    //             param.euler_param.ref_length,
    //             param.euler_param.gamma_gas,
    //             param.euler_param.mach_inf,
    //             param.euler_param.angle_of_attack,
    //             param.euler_param.side_slip_angle);

    std::string error_string;
    bool has_residual_converged = true;
    double artificial_dissipation_max_coeff = 0.0;
    double last_error=10;
    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    // Create initial condition function
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
    InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&param);

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {
        // p0 tends to require a finer grid to reach asymptotic region

        std::vector<double> grid_size(n_grids);
        //  density
        std::vector<double> error_L2_density(n_grids);
        std::vector<double> error_L1_density(n_grids);
        std::vector<double> error_Linf_density(n_grids);
        // pressure
        std::vector<double> error_L2_pressure(n_grids);
        std::vector<double> error_L1_pressure(n_grids);
        std::vector<double> error_Linf_pressure(n_grids);
        // temperature
        std::vector<double> error_L2_temperature(n_grids);
        std::vector<double> error_L1_temperature(n_grids);
        std::vector<double> error_Linf_temperature(n_grids);  
        // mass_fractions_1st
        std::vector<double> error_L2_mass_fractions_1st(n_grids);
        std::vector<double> error_L1_mass_fractions_1st(n_grids);
        std::vector<double> error_Linf_mass_fractions_1st(n_grids);              

        dealii::ConvergenceTable convergence_table;

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);
        // param.flow_solver_param.number_of_subdivisions_in_x_direction = n_1d_cells[0] * 4;
        // param.flow_solver_param.number_of_subdivisions_in_y_direction = n_1d_cells[0];

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            param.flow_solver_param.number_of_grid_elements_per_dimension = n_1d_cells[igrid];

            const double solution_degree = poly_degree;
            const double grid_degree = solution_degree+1;

            param.flow_solver_param.poly_degree = solution_degree;
            param.flow_solver_param.max_poly_degree_for_adaptation = solution_degree;
            param.flow_solver_param.grid_degree = grid_degree;
            param.flow_solver_param.number_of_mesh_refinements = igrid;

            pcout << "\n" << "************************************" << "\n" << "POLYNOMIAL DEGREE " << solution_degree 
                  << ", GRID NUMBER " << (igrid+1) << "/" << n_grids << "\n" << "************************************" << std::endl;

            pcout << "\n" << "Creating FlowSolver" << std::endl;

            std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);

            flow_solver->run();

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(flow_solver->dg->max_degree+1+overintegrate);
            const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
            dealii::FEValues<dim,dim> fe_values_extra(mapping, flow_solver->dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;
            // TO DO: define exact_at_q ... DONE
            std::array<double,nstate> exact_at_q;

            // density
            double l2error_density = 0;
            double l1error_density = 0;
            double linferror_density = 0;
            double linferror_density_candidate = 0;
            // pressure
            double l2error_pressure = 0;
            double l1error_pressure = 0;
            double linferror_pressure = 0;
            double linferror_pressure_candidate = 0;
            // temperature
            double l2error_temperature = 0;
            double l1error_temperature = 0;
            double linferror_temperature = 0;
            double linferror_temperature_candidate = 0; 
            // mass_fractions_1st
            double l2error_mass_fractions_1st = 0;
            double l1error_mass_fractions_1st = 0;
            double linferror_mass_fractions_1st = 0;
            double linferror_mass_fractions_1st_candidate = 0;                       

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

            // Integrate solution error and output error
            for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell!=flow_solver->dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    // TO DO: get exact_at_q here using x or qpoint ... DONE
                    for (unsigned int istate=0; istate<nstate; ++istate)
                    {
                        // Note: This is in non-dimensional form (free-stream values as reference)
                        if constexpr(dim == 1 || dim == 2)
                        {
                            // TO DO: get x here ... DONE
                            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                            
                            exact_at_q[istate] = flow_solver->flow_solver_case->initial_condition_function->value(qpoint,istate);
                        } else {
                            exact_at_q[istate] = 0.0;
                        }
                    }

                    // // THIS IS FOR 1/10 cycle
                    // for (unsigned int istate=0; istate<nstate; ++istate)
                    // {
                    //     // Note: This is in non-dimensional form (free-stream values as reference)
                    //     if constexpr(dim == 1 || dim == 2)
                    //     {
                    //         const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    //         exact_at_q[istate] = compute_exact_at_q(qpoint,istate);
                    //     } else {
                    //         exact_at_q[istate] = 0.0;
                    //     }
                    // }

                    double density_numerical, density_exact;
                    double pressure_numerical, pressure_exact;
                    double temperature_numerical, temperature_exact;          
                    double mass_fractions_1st_numerical, mass_fractions_1st_exact;                              
                    // if(param.artificial_dissipation_param.use_enthalpy_error)
                    // {
                    //     error_string = "L2_enthalpy_error";
                    //     const double pressure = euler_physics_double.compute_pressure(soln_at_q);
                    //     unumerical = euler_physics_double.compute_specific_enthalpy(soln_at_q,pressure);
                    //     uexact = euler_physics_double.gam*euler_physics_double.pressure_inf/euler_physics_double.density_inf*(1.0/euler_physics_double.gamm1+0.5*euler_physics_double.mach_inf_sqr);
                    // } 
                    // else
                    // {
                    //     error_string = "L2_entropy_error";
                    //     const double entropy_inf = euler_physics_double.entropy_inf;
                    //     unumerical = euler_physics_double.compute_entropy_measure(soln_at_q);
                    //     uexact = entropy_inf;
                    // }
                    error_string = "L2_density_error";

                    // Physics properties
                    // density
                    density_numerical = soln_at_q[0];
                    density_exact = exact_at_q[0];
                    // pressure
                    pressure_numerical = compute_pressure(soln_at_q);
                    pressure_exact =     compute_pressure(exact_at_q);
                    // temperature
                    temperature_numerical = compute_temperature(soln_at_q);
                    temperature_exact =     compute_temperature(exact_at_q);    
                    // mass_fractions_1st
                    mass_fractions_1st_numerical = compute_mass_fractions_1st(soln_at_q);
                    mass_fractions_1st_exact =     compute_mass_fractions_1st(exact_at_q);                                    
 
                    // error
                    // density
                    l2error_density += pow(density_numerical - density_exact, 2) * fe_values_extra.JxW(iquad);
                    l1error_density += abs(density_numerical - density_exact   ) * fe_values_extra.JxW(iquad);
                    linferror_density_candidate = abs(density_numerical - density_exact);
                    if(linferror_density_candidate > linferror_density) {linferror_density = linferror_density_candidate;}
                    // pressure
                    l2error_pressure += pow(pressure_numerical - pressure_exact, 2) * fe_values_extra.JxW(iquad);
                    l1error_pressure += abs(pressure_numerical - pressure_exact   ) * fe_values_extra.JxW(iquad);
                    linferror_pressure_candidate = abs(pressure_numerical - pressure_exact);
                    if(linferror_pressure_candidate > linferror_pressure) {linferror_pressure = linferror_pressure_candidate;}
                    // temperature
                    l2error_temperature += pow(temperature_numerical - temperature_exact, 2) * fe_values_extra.JxW(iquad);
                    l1error_temperature += abs(temperature_numerical - temperature_exact   ) * fe_values_extra.JxW(iquad);
                    linferror_temperature_candidate = abs(temperature_numerical - temperature_exact);
                    if(linferror_temperature_candidate > linferror_temperature) {linferror_temperature = linferror_temperature_candidate;}
                    // mass_fractions_1st
                    l2error_mass_fractions_1st += pow(mass_fractions_1st_numerical - mass_fractions_1st_exact, 2) * fe_values_extra.JxW(iquad);
                    l1error_mass_fractions_1st += abs(mass_fractions_1st_numerical - mass_fractions_1st_exact   ) * fe_values_extra.JxW(iquad);
                    linferror_mass_fractions_1st_candidate = abs(mass_fractions_1st_numerical - mass_fractions_1st_exact);
                    if(linferror_mass_fractions_1st_candidate > linferror_mass_fractions_1st) {linferror_mass_fractions_1st = linferror_mass_fractions_1st_candidate;}                                  
                }
            }
            // density
            const double l2error_density_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_density, mpi_communicator)); 
            const double l1error_density_mpi_sum =           dealii::Utilities::MPI::sum(l1error_density, mpi_communicator);
            const double linferror_density_mpi_max =         dealii::Utilities::MPI::max(linferror_density, mpi_communicator);
            // pressure
            const double l2error_pressure_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_pressure, mpi_communicator)); 
            const double l1error_pressure_mpi_sum =           dealii::Utilities::MPI::sum(l1error_pressure, mpi_communicator);
            const double linferror_pressure_mpi_max =         dealii::Utilities::MPI::max(linferror_pressure, mpi_communicator);
            // temperature
            const double l2error_temperature_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_temperature, mpi_communicator)); 
            const double l1error_temperature_mpi_sum =           dealii::Utilities::MPI::sum(l1error_temperature, mpi_communicator);
            const double linferror_temperature_mpi_max =         dealii::Utilities::MPI::max(linferror_temperature, mpi_communicator); 
            // mass_fractions_1st
            const double l2error_mass_fractions_1st_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_mass_fractions_1st, mpi_communicator)); 
            const double l1error_mass_fractions_1st_mpi_sum =           dealii::Utilities::MPI::sum(l1error_mass_fractions_1st, mpi_communicator);
            const double linferror_mass_fractions_1st_mpi_max =         dealii::Utilities::MPI::max(linferror_mass_fractions_1st, mpi_communicator);                       

            last_error = l2error_density_mpi_sum;

            const unsigned int n_dofs = flow_solver->dg->dof_handler.n_dofs();
            const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();

            // Convergence table
            double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            // density
            error_L2_density[igrid] = l2error_density_mpi_sum;
            error_L1_density[igrid] = l1error_density_mpi_sum;
            error_Linf_density[igrid] = linferror_density_mpi_max;
            // pressure
            error_L2_pressure[igrid] = l2error_pressure_mpi_sum;
            error_L1_pressure[igrid] = l1error_pressure_mpi_sum;
            error_Linf_pressure[igrid] = linferror_pressure_mpi_max;
            // temperature
            error_L2_temperature[igrid] = l2error_temperature_mpi_sum;
            error_L1_temperature[igrid] = l1error_temperature_mpi_sum;
            error_Linf_temperature[igrid] = linferror_temperature_mpi_max;           
            // mass_fractions_1st
            error_L2_mass_fractions_1st[igrid] = l2error_mass_fractions_1st_mpi_sum;
            error_L1_mass_fractions_1st[igrid] = l1error_mass_fractions_1st_mpi_sum;
            error_Linf_mass_fractions_1st[igrid] = linferror_mass_fractions_1st_mpi_max;             

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("L1_error(density)", l1error_density_mpi_sum);
            convergence_table.add_value("L2_error(density)", l2error_density_mpi_sum);
            convergence_table.add_value("Linf_error(density)", linferror_density_mpi_max);
            convergence_table.add_value("L1_error(pressure)", l1error_pressure_mpi_sum);
            convergence_table.add_value("L2_error(pressure)", l2error_pressure_mpi_sum);
            convergence_table.add_value("Linf_error(pressure)", linferror_pressure_mpi_max);
            convergence_table.add_value("L1_error(temperature)", l1error_temperature_mpi_sum);
            convergence_table.add_value("L2_error(temperature)", l2error_temperature_mpi_sum);
            convergence_table.add_value("Linf_error(temperature)", linferror_temperature_mpi_max);   
            convergence_table.add_value("L1_error(species #1)", l1error_mass_fractions_1st_mpi_sum);
            convergence_table.add_value("L2_error(species #1)", l2error_mass_fractions_1st_mpi_sum);
            convergence_table.add_value("Linf_error(species #1)", linferror_mass_fractions_1st_mpi_max);                     
            convergence_table.add_value("Residual",flow_solver->ode_solver->residual_norm);
            
            if(flow_solver->ode_solver->residual_norm > 1e-10)
            {
                has_residual_converged = false;
            }

            artificial_dissipation_max_coeff = flow_solver->dg->max_artificial_dissipation_coeff;

            // density
            pcout << "Density Error \n"
                 << " Grid size h: " << dx 
                 << " L1-error: " << l1error_density_mpi_sum
                 << " L2-error: " << l2error_density_mpi_sum
                 << " Linf-error: " << linferror_density_mpi_max
                 << " Residual: " << flow_solver->ode_solver->residual_norm
                 << std::endl;

            // pressure
            pcout << "Pressure Error \n"
                 << " Grid size h: " << dx 
                 << " L1-error: " << l1error_pressure_mpi_sum
                 << " L2-error: " << l2error_pressure_mpi_sum
                 << " Linf-error: " << linferror_pressure_mpi_max
                 << " Residual: " << flow_solver->ode_solver->residual_norm
                 << std::endl;

            // temperature
            pcout << "Temperature Error \n"
                 << " Grid size h: " << dx 
                 << " L1-error: " << l1error_temperature_mpi_sum
                 << " L2-error: " << l2error_temperature_mpi_sum
                 << " Linf-error: " << linferror_temperature_mpi_max
                 << " Residual: " << flow_solver->ode_solver->residual_norm
                 << std::endl;  

            // mass_fractions_1st
            pcout << "Species #1 Error \n"
                 << " Grid size h: " << dx 
                 << " L1-error: " << l1error_mass_fractions_1st_mpi_sum
                 << " L2-error: " << l2error_mass_fractions_1st_mpi_sum
                 << " Linf-error: " << linferror_mass_fractions_1st_mpi_max
                 << " Residual: " << flow_solver->ode_solver->residual_norm
                 << std::endl;                                

            // file output
            // density
            outdata_density << dx << " " 
                    << l1error_density_mpi_sum << " "
                    << l2error_density_mpi_sum << " "
                    << linferror_density_mpi_max << " "
                    << std::endl;
            // pressure
            outdata_pressure << dx << " " 
                    << l1error_pressure_mpi_sum << " "
                    << l2error_pressure_mpi_sum << " "
                    << linferror_pressure_mpi_max << " "
                    << std::endl;
            // temperature
            outdata_temperature << dx << " " 
                    << l1error_temperature_mpi_sum << " "
                    << l2error_temperature_mpi_sum << " "
                    << linferror_temperature_mpi_max << " "
                    << std::endl;      
            // mass_fractions_1st
            outdata_mass_fractions_1st << dx << " " 
                    << l1error_mass_fractions_1st_mpi_sum << " "
                    << l2error_mass_fractions_1st_mpi_sum << " "
                    << linferror_mass_fractions_1st_mpi_max << " "
                    << std::endl;                                  

            if (igrid > 0) {
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl;

                // density
                double slope_soln_err_l1 = log(error_L1_density[igrid]/error_L1_density[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                double slope_soln_err_l2 = log(error_L2_density[igrid]/error_L2_density[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                double slope_soln_err_linf = log(error_Linf_density[igrid]/error_Linf_density[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout <<"Density Slope \n"           
                     <<" " << "L1_error " << 1 << "  " << error_L1_density[igrid-1]
                     <<" " << "L1_error " << 2 << "  " << error_L1_density[igrid]
                     << "  slope " << slope_soln_err_l1
                     << std::endl
                     <<" " << "L2_error " << 1 << "  " << error_L2_density[igrid-1]
                     <<" " << "L2_error " << 2 << "  " << error_L2_density[igrid]
                     << "  slope " << slope_soln_err_l2
                     << std::endl
                     <<" " << "Linf_error " << 1 << "  " << error_Linf_density[igrid-1]
                     <<" " << "Linf_error " << 2 << "  " << error_Linf_density[igrid]
                     << "  slope " << slope_soln_err_linf
                     << std::endl;
                // file output
                outdata_slope_density << dx << " " 
                    << slope_soln_err_l1 << " "
                    << slope_soln_err_l2 << " "
                    << slope_soln_err_linf << " "
                    << std::endl;     
                // pressure
                slope_soln_err_l1 = log(error_L1_pressure[igrid]/error_L1_pressure[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_l2 = log(error_L2_pressure[igrid]/error_L2_pressure[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_linf = log(error_Linf_pressure[igrid]/error_Linf_pressure[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout <<"Pressure Slope \n"           
                     <<" " << "L1_error " << 1 << "  " << error_L1_pressure[igrid-1]
                     <<" " << "L1_error " << 2 << "  " << error_L1_pressure[igrid]
                     << "  slope " << slope_soln_err_l1
                     << std::endl
                     <<" " << "L2_error " << 1 << "  " << error_L2_pressure[igrid-1]
                     <<" " << "L2_error " << 2 << "  " << error_L2_pressure[igrid]
                     << "  slope " << slope_soln_err_l2
                     << std::endl
                     <<" " << "Linf_error " << 1 << "  " << error_Linf_pressure[igrid-1]
                     <<" " << "Linf_error " << 2 << "  " << error_Linf_pressure[igrid]
                     << "  slope " << slope_soln_err_linf
                     << std::endl;
                // file output
                outdata_slope_pressure << dx << " " 
                    << slope_soln_err_l1 << " "
                    << slope_soln_err_l2 << " "
                    << slope_soln_err_linf << " "
                    << std::endl;        
                // temperature
                slope_soln_err_l1 = log(error_L1_temperature[igrid]/error_L1_temperature[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_l2 = log(error_L2_temperature[igrid]/error_L2_temperature[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_linf = log(error_Linf_temperature[igrid]/error_Linf_temperature[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout <<"Temperature Slope \n"           
                     <<" " << "L1_error " << 1 << "  " << error_L1_temperature[igrid-1]
                     <<" " << "L1_error " << 2 << "  " << error_L1_temperature[igrid]
                     << "  slope " << slope_soln_err_l1
                     << std::endl
                     <<" " << "L2_error " << 1 << "  " << error_L2_temperature[igrid-1]
                     <<" " << "L2_error " << 2 << "  " << error_L2_temperature[igrid]
                     << "  slope " << slope_soln_err_l2
                     << std::endl
                     <<" " << "Linf_error " << 1 << "  " << error_Linf_temperature[igrid-1]
                     <<" " << "Linf_error " << 2 << "  " << error_Linf_temperature[igrid]
                     << "  slope " << slope_soln_err_linf
                     << std::endl;     
                // file output
                outdata_slope_temperature << dx << " " 
                    << slope_soln_err_l1 << " "
                    << slope_soln_err_l2 << " "
                    << slope_soln_err_linf << " "
                    << std::endl;                      
                // mass_fractions_1st
                slope_soln_err_l1 = log(error_L1_mass_fractions_1st[igrid]/error_L1_mass_fractions_1st[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_l2 = log(error_L2_mass_fractions_1st[igrid]/error_L2_mass_fractions_1st[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_soln_err_linf = log(error_Linf_mass_fractions_1st[igrid]/error_Linf_mass_fractions_1st[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout <<"Species #1 Slope \n"           
                     <<" " << "L1_error " << 1 << "  " << error_L1_mass_fractions_1st[igrid-1]
                     <<" " << "L1_error " << 2 << "  " << error_L1_mass_fractions_1st[igrid]
                     << "  slope " << slope_soln_err_l1
                     << std::endl
                     <<" " << "L2_error " << 1 << "  " << error_L2_mass_fractions_1st[igrid-1]
                     <<" " << "L2_error " << 2 << "  " << error_L2_mass_fractions_1st[igrid]
                     << "  slope " << slope_soln_err_l2
                     << std::endl
                     <<" " << "Linf_error " << 1 << "  " << error_Linf_mass_fractions_1st[igrid-1]
                     <<" " << "Linf_error " << 2 << "  " << error_Linf_mass_fractions_1st[igrid]
                     << "  slope " << slope_soln_err_linf
                     << std::endl;
                // file output
                outdata_slope_mass_fractions_1st << dx << " " 
                    << slope_soln_err_l1 << " "
                    << slope_soln_err_l2 << " "
                    << slope_soln_err_linf << " "  
                    << std::endl;                                                                             
            }

            //output_results (igrid);
            if (igrid == n_grids-1)
            {
                if (param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0)
                {
                    dealii::Point<dim> smallest_cell_coord = flow_solver->dg->coordinates_of_highest_refined_cell();
                    pcout<<" x = "<<smallest_cell_coord[0]<<" y = "<<smallest_cell_coord[1]<<std::endl;
                    // Check if the mesh is refined near the shock i.e x \in (0.1,0.5) and y \in (0.0, 0.5).
                    if ((smallest_cell_coord[0] > 0.1) && (smallest_cell_coord[0] < 0.5) && (smallest_cell_coord[1] > 0.0) && (smallest_cell_coord[1] < 0.5)) 
                    {
                        pcout<<"Mesh is refined near the shock. Test passed"<<std::endl;
                        return 0; // Mesh adaptation test passed.
                    }
                    return 1; // Mesh adaptation failed.
                }
            }
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates(error_string, "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific(error_string, true);
        convergence_table.set_scientific("Residual",true);
        //convergence_table.set_scientific("L2_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;
    
        double last_slope = 0.0;

        if(n_grids>=2)
        {
            last_slope = log(error_L2_density[n_grids-1]/error_L2_density[n_grids-2]) / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        }

        const double slope_avg = last_slope;
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if( (slope_diff < slope_deficit_tolerance) || (n_grids==1) ) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

//****************Test for artificial dissipation begins *******************************************************
    using artificial_dissipation_test_enum = Parameters::ArtificialDissipationParam::ArtificialDissipationTestType;
    artificial_dissipation_test_enum arti_dissipation_test_type = param.artificial_dissipation_param.artificial_dissipation_test_type;
    if (arti_dissipation_test_type == artificial_dissipation_test_enum::residual_convergence)
    {
        if(has_residual_converged)
        {
           pcout << std::endl << "Residual has converged. Test Passed"<<std::endl;
            return 0;
        }
        pcout << std::endl<<"Residual has not converged. Test failed" << std::endl;
        return 1;
    }
    else if (arti_dissipation_test_type == artificial_dissipation_test_enum::discontinuity_sensor_activation) 
    {
        if(artificial_dissipation_max_coeff < 1e-10)
        {
            pcout << std::endl << "Discontinuity sensor is not activated. Max dissipation coeff = " <<artificial_dissipation_max_coeff << "   Test passed"<<std::endl;
            return 0;
        }
        pcout << std::endl << "Discontinuity sensor has been activated. Max dissipation coeff = " <<artificial_dissipation_max_coeff << "   Test failed"<<std::endl;
        return 1;
    }
    else if (arti_dissipation_test_type == artificial_dissipation_test_enum::enthalpy_conservation) 
    {
        return last_error; // Return the error
    }
//****************Test for artificial dissipation ends *******************************************************
    else
    {
        int n_fail_poly = fail_conv_poly.size();
        if (n_fail_poly > 0) 
        {
            for (int ifail=0; ifail < n_fail_poly; ++ifail) 
            {
                const double expected_slope = fail_conv_poly[ifail]+1;
                const double slope_deficit_tolerance = -0.1;
                pcout << std::endl
                     << "Convergence order not achieved for polynomial p = "
                     << fail_conv_poly[ifail]
                     << ". Slope of "
                     << fail_conv_slop[ifail] << " instead of expected "
                     << expected_slope << " within a tolerance of "
                     << slope_deficit_tolerance
                     << std::endl;
            }
        }
        return n_fail_poly;
    }
}


// #if PHILIP_DIM==2
//     template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2>;
// #endif

// #if PHILIP_DIM==1
// template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM>;
template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2>; // euler (1-species)_
#if PHILIP_DIM==1
template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2+2-1>; // TO DO: N_SPECIES
#endif
#if PHILIP_DIM==2
template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2+2-1>; // TO DO: N_SPECIES
#endif
// #else
// template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2>;
// template class EulerVortexAdvectionErrorStudy <PHILIP_DIM,PHILIP_DIM+2+2-1>; // TO DO: N_SPECIES
// #endif

} // Tests namespace
} // PHiLiP namespace

