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
Parameters::AllParameters TaylorGreenVortexRestartCheck<dim, nstate>::reinit_params(
    const bool output_restart_files_input,
    const bool restart_computation_from_file_input,
    const double final_time_input,
    const double initial_time_input,
    const unsigned int initial_iteration_input,
    const double initial_desired_time_for_output_solution_every_dt_time_intervals_input) const {

    // copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    // change desired parameters based on inputs
    parameters.flow_solver_param.output_restart_files = output_restart_files_input;
    parameters.flow_solver_param.restart_computation_from_file = restart_computation_from_file_input;
    parameters.flow_solver_param.final_time = final_time_input;
    parameters.ode_solver_param.initial_time = initial_time_input;
    parameters.ode_solver_param.initial_iteration = initial_iteration_input;
    parameters.ode_solver_param.initial_desired_time_for_output_solution_every_dt_time_intervals = initial_desired_time_for_output_solution_every_dt_time_intervals_input;

    return parameters;
}

template <int dim, int nstate>
int TaylorGreenVortexRestartCheck<dim, nstate>::run_test() const
{
    const double time_at_which_we_stop_the_run = 6.1240484302437529e-03;
    // corresponds to the expected kinetic energy set in the prm file
    const double time_at_which_the_run_is_complete = this->all_parameters->flow_solver_param.final_time; // TO DO: Update without this hack
    Parameters::AllParameters params_incomplete_run = reinit_params(true,false,time_at_which_we_stop_the_run);
    Parameters::AllParameters params_restart_to_complete_run = reinit_params(false,true,time_at_which_the_run_is_complete,time_at_which_we_stop_the_run,5,0.0); // TO DO: update the last arg

    // Integrate to time at which we stop the run
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_incomplete_run = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_incomplete_run);
    static_cast<void>(flow_solver_incomplete_run->run_test());

    // Integrate to final time by restarting from where we stopped
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_restart_to_complete_run = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_restart_to_complete_run);
    static_cast<void>(flow_solver_restart_to_complete_run->run_test());

    // Compute kinetic energy at final time achieved by restarting the computation
    std::unique_ptr<PeriodicCubeFlow<dim, nstate>> flow_solver_case = std::make_unique<PeriodicCubeFlow<dim,nstate>>(this->all_parameters);
    const double kinetic_energy_computed = flow_solver_case->compute_kinetic_energy(*(flow_solver_restart_to_complete_run->dg));

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