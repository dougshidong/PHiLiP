#include "rrk_numerical_entropy_conservation_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"    
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"    
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
RRKNumericalEntropyConservationCheck<dim, nstate>::RRKNumericalEntropyConservationCheck(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters RRKNumericalEntropyConservationCheck<dim,nstate>::reinit_params(bool use_rrk, double time_step_size_factor) const
{
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
    
    parameters.ode_solver_param.initial_time_step*=time_step_size_factor;
    parameters.flow_solver_param.courant_friedrichs_lewy_number*=time_step_size_factor;
    
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    if (use_rrk)    {parameters.ode_solver_param.ode_solver_type = ODESolverEnum::rrk_explicit_solver;}
    else            {parameters.ode_solver_param.ode_solver_type = ODESolverEnum::runge_kutta_solver;}

    return parameters;
}

template <int dim, int nstate>
int RRKNumericalEntropyConservationCheck<dim, nstate>::compare_numerical_entropy_to_initial(
        const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> &flow_solver,
        const double initial_numerical_entropy,
        const double final_time_actual,
        bool expect_conservation
        ) const{
    
    //pointer to flow_solver_case for computing numerical_entropy
    // Using PHILIP_DIM as an indicator for whether the test is using Burgers, in
    // which case we want to use Periodic1DUnsteady, or Euler, in which case we
    // want to use PeriodicTurbulence
#if PHILIP_DIM==1
    std::shared_ptr<FlowSolver::Periodic1DUnsteady<dim, nstate>> flow_solver_case = std::dynamic_pointer_cast<FlowSolver::Periodic1DUnsteady<dim, nstate>>(flow_solver->flow_solver_case);
#else
    std::shared_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::dynamic_pointer_cast<FlowSolver::PeriodicTurbulence<dim, nstate>>(flow_solver->flow_solver_case);
#endif

    const double final_numerical_entropy = flow_solver_case->get_numerical_entropy(flow_solver->dg);
    const double numerical_entropy_change = abs(initial_numerical_entropy-final_numerical_entropy);
    pcout << "At end time t = " << final_time_actual << ", numerical entropy change at end was " << std::fixed << std::setprecision(16) << numerical_entropy_change <<std::endl;
    if (expect_conservation && (numerical_entropy_change< 1E-13)){
        pcout << "Numerical entropy was conserved, as expected." << std::endl;
        return 0; //pass test
    } else if (!expect_conservation && (numerical_entropy_change > 1E-13)){
        pcout << "Numerical entropy was NOT conserved, as expected." << std::endl;
        return 0; //pass test
    }else if (expect_conservation && (numerical_entropy_change > 1E-13)){
        pcout << "Numerical entropy was NOT conserved, but was expected to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        return 1; //fail test
    }else{
        pcout << "Numerical entropy was conserved, but was expected NOT to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        return 1; //fail test (included for completeness, but not expected to be used)
    }
}

template <int dim, int nstate>
int RRKNumericalEntropyConservationCheck<dim,nstate>::get_numerical_entropy_and_compare_to_initial(
        const Parameters::AllParameters params,
        const double numerical_entropy_initial,
        bool expect_conservation
        ) const
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case(&params, parameter_handler);
    static_cast<void>(flow_solver->run());
    const double final_time_actual = flow_solver->ode_solver->current_time;
    int failed_this_calculation = compare_numerical_entropy_to_initial(flow_solver, numerical_entropy_initial, final_time_actual, expect_conservation);
    return failed_this_calculation;
}

template <int dim, int nstate>
int RRKNumericalEntropyConservationCheck<dim, nstate>::run_test() const
{

    double final_time = this->all_parameters->flow_solver_param.final_time;
    double time_step_large = this->all_parameters->ode_solver_param.initial_time_step;
    double time_step_reduction_factor = 1E-2;

    int n_steps = round(final_time/time_step_large);
    if (n_steps * time_step_large!= final_time){
        pcout << "WARNING: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, time_step_large)
              << ". Consider modifying parameters." << std::endl;
    }

    int testfail = 0;
    int failed_this_calculation = 0;
    
    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Calculating initial numerical entropy..." << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case((this->all_parameters), parameter_handler);

    // Using PHILIP_DIM as an indicator for whether the test is using Burgers, in
    // which case we want to use Periodic1DUnsteady, or Euler, in which case we
    // want to use PeriodicTurbulence
#if PHILIP_DIM==1
    std::shared_ptr<FlowSolver::Periodic1DUnsteady<dim, nstate>> flow_solver_case = std::dynamic_pointer_cast<FlowSolver::Periodic1DUnsteady<dim, nstate>>(flow_solver->flow_solver_case);
#else
    std::shared_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::dynamic_pointer_cast<FlowSolver::PeriodicTurbulence<dim, nstate>>(flow_solver->flow_solver_case);
#endif
    
    const double numerical_entropy_initial = flow_solver_case->get_numerical_entropy(flow_solver->dg); //no need to run as ode_solver is allocated during construction
    pcout << "   Initial numerical_entropy : " << numerical_entropy_initial << std::endl;

    // Run four main tests
    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_rrk = reinit_params(true, 1.0);
    failed_this_calculation = get_numerical_entropy_and_compare_to_initial(params_large_rrk,
                                              numerical_entropy_initial,
                                              true); //expect_conservation = true
    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_norrk = reinit_params(false, 1.0);
    failed_this_calculation = get_numerical_entropy_and_compare_to_initial(params_large_norrk,
                                              numerical_entropy_initial,
                                              false); //expect_conservation = false 
    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, reducing by " << time_step_reduction_factor << " and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_small_rrk = reinit_params(true, time_step_reduction_factor);
    failed_this_calculation = get_numerical_entropy_and_compare_to_initial(params_small_rrk,
                                              numerical_entropy_initial,
                                              true); //expect_conservation = true
    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, reducing by " << time_step_reduction_factor << " without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    
    const Parameters::AllParameters params_small_norrk = reinit_params(false, time_step_reduction_factor);
    failed_this_calculation = get_numerical_entropy_and_compare_to_initial(params_small_norrk,
                                              numerical_entropy_initial,
                                              true); //expect_conservation = true
    if (failed_this_calculation) testfail = 1;

    return testfail;
}
#if PHILIP_DIM == 1
    template class RRKNumericalEntropyConservationCheck<PHILIP_DIM,PHILIP_DIM>;
#elif PHILIP_DIM == 3
    template class RRKNumericalEntropyConservationCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
