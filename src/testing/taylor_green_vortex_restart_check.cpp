#include "taylor_green_vortex_restart_check.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_cube_flow.h"
#include <deal.II/base/table_handler.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TaylorGreenVortexRestartCheck<dim, nstate>::TaylorGreenVortexRestartCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template <int dim, int nstate>
Parameters::AllParameters TaylorGreenVortexRestartCheck<dim, nstate>::reinit_params(
    const bool output_restart_files_input,
    const bool restart_computation_from_file_input,
    const double final_time_input,
    const double initial_time_input,
    const unsigned int initial_iteration_input,
    const double initial_desired_time_for_output_solution_every_dt_time_intervals_input,
    const int restart_file_index_input) const {

    // copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    // change desired parameters based on inputs
    parameters.flow_solver_param.output_restart_files = output_restart_files_input;
    parameters.flow_solver_param.restart_computation_from_file = restart_computation_from_file_input;
    parameters.flow_solver_param.final_time = final_time_input;
    parameters.ode_solver_param.initial_time = initial_time_input;
    parameters.ode_solver_param.initial_iteration = initial_iteration_input;
    parameters.ode_solver_param.initial_desired_time_for_output_solution_every_dt_time_intervals = initial_desired_time_for_output_solution_every_dt_time_intervals_input;
    parameters.flow_solver_param.restart_file_index = restart_file_index_input;

    return parameters;
}

template<typename InputIterator1, typename InputIterator2>
bool range_equal(InputIterator1 first1, InputIterator1 last1,
        InputIterator2 first2, InputIterator2 last2)
{
    // Code reference: https://stackoverflow.com/questions/15118661/in-c-whats-the-fastest-way-to-tell-whether-two-string-or-binary-files-are-di
    while(first1 != last1 && first2 != last2)
    {
        if(*first1 != *first2) return false;
        ++first1;
        ++first2;
    }
    return (first1 == last1) && (first2 == last2);
}

bool compare_files(const std::string& filename1, const std::string& filename2)
{
    // Code reference: https://stackoverflow.com/questions/15118661/in-c-whats-the-fastest-way-to-tell-whether-two-string-or-binary-files-are-di
    std::ifstream file1(filename1);
    std::ifstream file2(filename2);

    std::istreambuf_iterator<char> begin1(file1);
    std::istreambuf_iterator<char> begin2(file2);

    std::istreambuf_iterator<char> end;

    return range_equal(begin1, end, begin2, end);
}

template <int dim, int nstate>
int TaylorGreenVortexRestartCheck<dim, nstate>::run_test() const
{
    const double time_at_which_we_stop_the_run = 6.1240484302437529e-03;
    const int restart_file_index = 4;
    const int initial_iteration_restart = restart_file_index; // assumes output mod for restart files is 1
    const double time_at_which_the_run_is_complete = this->all_parameters->flow_solver_param.final_time;
    Parameters::AllParameters params_incomplete_run = reinit_params(true,false,time_at_which_we_stop_the_run);
    Parameters::AllParameters params_restart_to_complete_run = reinit_params(false,true,time_at_which_the_run_is_complete,time_at_which_we_stop_the_run,initial_iteration_restart,0.0,restart_file_index);

    // Integrate to time at which we stop the run
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_incomplete_run = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_incomplete_run, parameter_handler);
    static_cast<void>(flow_solver_incomplete_run->run_test());

    // INLINE SUB-TEST: Check whether the initialize_data_table_from_file() function in flow solver is working correctly
    if(this->mpi_rank==0) {
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
        const std::string file_read = params_incomplete_run.flow_solver_param.restart_files_directory_name+std::string("/")+params_incomplete_run.flow_solver_param.unsteady_data_table_filename+std::string("-")+flow_solver_incomplete_run->get_restart_filename_without_extension(restart_file_index)+std::string(".txt");
        flow_solver_incomplete_run->initialize_data_table_from_file(file_read,unsteady_data_table);
        const std::string file_write = "read_table_check.txt";
        std::ofstream unsteady_data_table_file(file_write);
        unsteady_data_table->write_text(unsteady_data_table_file);
        // check if files are the same (i.e. if the tables are the same)
        bool files_are_same = compare_files(file_read,file_write);
        if(!files_are_same) {
            pcout << "\n Error: initialize_data_table_from_file() failed." << std::endl;
            return 1;
        }
        else {
            pcout << "\n Sub-test for initialize_data_table_from_file() passed, continuing test..." << std::endl;
        }
    } // END

    // Integrate to final time by restarting from where we stopped
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_restart_to_complete_run = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_restart_to_complete_run, parameter_handler);
    static_cast<void>(flow_solver_restart_to_complete_run->run_test());

    // Compute kinetic energy at final time achieved by restarting the computation
    std::unique_ptr<PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<PeriodicTurbulence<dim,nstate>>(this->all_parameters);
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