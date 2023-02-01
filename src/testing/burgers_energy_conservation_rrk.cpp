#include "burgers_energy_conservation_rrk.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"    
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersEnergyConservationRRK<dim, nstate>::BurgersEnergyConservationRRK(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters BurgersEnergyConservationRRK<dim,nstate>::reinit_params(bool use_rrk, double time_step_size) const
{
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
    
    parameters.ode_solver_param.initial_time_step = time_step_size;
    
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    if (use_rrk)    {parameters.ode_solver_param.ode_solver_type = ODESolverEnum::rrk_explicit_solver;}
    else            {parameters.ode_solver_param.ode_solver_type = ODESolverEnum::runge_kutta_solver;}

    return parameters;
}

template <int dim, int nstate>
int BurgersEnergyConservationRRK<dim, nstate>::compare_energy_to_initial(
        const std::shared_ptr <DGBase<dim, double>> dg,
        const double initial_energy,
        bool expect_conservation
        ) const{
    
    //pointer to flow_solver_case for computing energy
    std::unique_ptr<FlowSolver::Periodic1DUnsteady<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::Periodic1DUnsteady<dim,nstate>>(this->all_parameters);

    const double final_energy = flow_solver_case->compute_energy_collocated(dg);
    const double energy_change = abs(initial_energy-final_energy);
    pcout << "Energy change at end was " << energy_change <<std::endl;
    if (expect_conservation && (energy_change< 1E-13)){
        pcout << "Energy was conserved, as expected." << std::endl;
        return 0; //pass test
    } else if (!expect_conservation && (energy_change > 1E-13)){
        pcout << "Energy was NOT conserved, as expected." << std::endl;
        return 0; //pass test
    }else if (expect_conservation && (energy_change > 1E-13)){
        pcout << "Energy was NOT conserved, but was expected to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        return 1; //fail test
    }else{
        pcout << "Energy was conserved, but was expected NOT to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        return 1; //fail test (included for completeness, but not expected to be used)
    }
}

template <int dim, int nstate>
int BurgersEnergyConservationRRK<dim,nstate>::get_energy_and_compare_to_initial(
        const Parameters::AllParameters params,
        const double energy_initial,
        bool expect_conservation
        ) const
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
    static_cast<void>(flow_solver->run());
    int failed_this_calculation = compare_energy_to_initial(flow_solver->dg, energy_initial, expect_conservation);
    return failed_this_calculation;
}

template <int dim, int nstate>
int BurgersEnergyConservationRRK<dim, nstate>::run_test() const
{

    double final_time = this->all_parameters->flow_solver_param.final_time;
    double time_step_large = this->all_parameters->ode_solver_param.initial_time_step;
    double time_step_small = time_step_large * 1E-2;

    int n_steps = round(final_time/time_step_large);
    if (n_steps * time_step_large != final_time){
        pcout << "Error: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, time_step_large)
              << ". Modify parameters to run this test." << std::endl;
        std::abort();
    }

    int testfail = 0;
    int failed_this_calculation = 0;
    
    //pointer to flow_solver_case for computing energy
    std::unique_ptr<FlowSolver::Periodic1DUnsteady<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::Periodic1DUnsteady<dim,nstate>>(this->all_parameters);

    // Get initial energy
    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Calculating initial energy..." << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case((this->all_parameters), parameter_handler);
    const double energy_initial = flow_solver_case->compute_energy_collocated(flow_solver->dg); //no need to run as ode_solver is allocated during construction
    pcout << "   Initial energy : " << energy_initial << std::endl;

    // Run four main tests
    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep, dt = " << time_step_large << " and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_rrk = reinit_params(true, time_step_large);
    failed_this_calculation = get_energy_and_compare_to_initial(params_large_rrk,
                                              energy_initial,
                                              true); //expect_conservation = true

    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep, dt = " << time_step_large << " without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_norrk = reinit_params(false, time_step_large);
    failed_this_calculation = get_energy_and_compare_to_initial(params_large_norrk,
                                              energy_initial,
                                              false); //expect_conservation = false 
    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, dt = " << time_step_small << " and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_small_rrk = reinit_params(true, time_step_small);
    failed_this_calculation = get_energy_and_compare_to_initial(params_small_rrk,
                                              energy_initial,
                                              true); //expect_conservation = true
    if (failed_this_calculation) testfail = 1;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, dt = " << time_step_small << " without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    
    const Parameters::AllParameters params_small_norrk = reinit_params(false, time_step_small);
    failed_this_calculation = get_energy_and_compare_to_initial(params_small_norrk,
                                              energy_initial,
                                              true); //expect_conservation = true
    if (failed_this_calculation) testfail = 1;

    return testfail;
}

#if PHILIP_DIM==1
    template class BurgersEnergyConservationRRK<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
