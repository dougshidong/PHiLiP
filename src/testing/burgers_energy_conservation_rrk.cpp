#include "burgers_energy_conservation_rrk.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_1D_unsteady.h"
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
     parameters.ode_solver_param.relaxation_runge_kutta = use_rrk;

     if (time_step_size < 1E-9){
        //for calculating initial energy - only want to take 1 timestep        
        parameters.flow_solver_param.final_time = time_step_size;
     }
     
     return parameters;
}

template <int dim, int nstate>
int BurgersEnergyConservationRRK<dim, nstate>::compare_energy_to_initial(
        const std::shared_ptr <DGBase<dim, double>> dg,
        const double initial_energy,
        bool expect_conservation
        ) const{
    
    const double final_energy = compute_energy_collocated(dg);
    const double energy_change = abs(initial_energy-final_energy);
    if (expect_conservation && (energy_change< 1E-13)){
        pcout << "Energy was conserved, as expected." << std::endl;
        pcout << "    Energy change at end was " << energy_change <<std::endl;
        return 0; //pass test
    } else if (!expect_conservation && (energy_change > 1E-13)){
        pcout << "Energy was NOT conserved, as expected." << std::endl;
        pcout << "    Energy change at end was " << energy_change <<std::endl;
        return 0; //pass test
    }else if (expect_conservation && (energy_change > 1E-13)){
        pcout << "Energy was NOT conserved, but was expected to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        pcout << "    Energy change at end was " << energy_change <<std::endl;
        return 1; //fail test
    }else{
        pcout << "Energy was conserved, but was expected NOT to be conserved." << std::endl;
        pcout << "    Unexpected result! Test failing." << std::endl;
        pcout << "    Energy change at end was " << energy_change <<std::endl;
        return 1; //fail test
    }
}

template <int dim, int nstate>
double BurgersEnergyConservationRRK<dim, nstate>::compute_energy_collocated(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    double energy = 0.0;
    for (unsigned int i = 0; i < dg->solution.size(); ++i)
    {
        energy += 1./(dg->global_inverse_mass_matrix.diag_element(i)) * dg->solution(i) * dg->solution(i);
    }
    return energy;
}

template <int dim, int nstate>
int BurgersEnergyConservationRRK<dim, nstate>::run_test() const
{

    double final_time = this->all_parameters->flow_solver_param.final_time;
    double time_step_large = this->all_parameters->ode_solver_param.initial_time_step;
    double time_step_small = time_step_large * 10E-3;

    int n_steps = round(final_time/time_step_large);
    if (n_steps * time_step_large != final_time){
        pcout << "Error: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, time_step_large)
              << ". Modify parameters to run this test." << std::endl;
        std::abort();
    }

    int testfail = 0;

    
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver;

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using very small timestep dt = " << 1E-10 << " for initial energy" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    //take a very small step to calculate initial energy
    //necessary because the inverse mass matrix is not initialized until timestepping starts
    const Parameters::AllParameters params_initial = reinit_params(true, 1E-10);
    //params_initial.flow_solver_param.final_time = 1E-8;
    flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_initial, parameter_handler);
    static_cast<void>(flow_solver->run_test());
    const double energy_initial = compute_energy_collocated(flow_solver->dg);

    //Run four main tests
    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep, dt = " << time_step_large << " and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_rrk = reinit_params(true, time_step_large);
    flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_large_rrk, parameter_handler);
    static_cast<void>(flow_solver->run_test());
    testfail = compare_energy_to_initial(flow_solver->dg, energy_initial, true); //expect_conservation = true

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using large timestep, dt = " << time_step_large << " without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_large_norrk = reinit_params(false, time_step_large);
    flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_large_norrk, parameter_handler);
    static_cast<void>(flow_solver->run_test());
    testfail = compare_energy_to_initial(flow_solver->dg, energy_initial, false); //expect_conservation = false

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, dt = " << time_step_small << " and RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;

    const Parameters::AllParameters params_small_rrk = reinit_params(true, time_step_small);
    flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_small_rrk, parameter_handler);
    static_cast<void>(flow_solver->run_test());
    testfail = compare_energy_to_initial(flow_solver->dg, energy_initial, true); //expect_conservation = true

    pcout << "\n\n-------------------------------------------------------------" << std::endl;
    pcout << "  Using small timestep, dt = " << time_step_small << " without RRK" << std::endl;
    pcout << "-------------------------------------------------------------" << std::endl;
    
    const Parameters::AllParameters params_small_norrk = reinit_params(false, time_step_small);
    flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_small_norrk, parameter_handler);
    static_cast<void>(flow_solver->run_test());
    testfail = compare_energy_to_initial(flow_solver->dg, energy_initial, true); //expect_conservation = true

    return testfail;
}

#if PHILIP_DIM==1
    template class BurgersEnergyConservationRRK<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
