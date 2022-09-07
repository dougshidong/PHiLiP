#include "decaying_homogeneous_isotropic_turbulence_init_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"
#include <deal.II/base/table_handler.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DecayingHomogeneousIsotropicTurbulenceInitCheck<dim, nstate>::DecayingHomogeneousIsotropicTurbulenceInitCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template <int dim, int nstate>
Parameters::AllParameters DecayingHomogeneousIsotropicTurbulenceInitCheck<dim, nstate>::reinit_params(
    const bool output_restart_files_input,
    const bool restart_computation_from_file_input,
    const double final_time_input,
    const double initial_time_input,
    const unsigned int initial_iteration_input,
    const double initial_desired_time_for_output_solution_every_dt_time_intervals_input,
    const double initial_time_step_input,
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
    parameters.ode_solver_param.initial_time_step = initial_time_step_input;
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

void add_value_to_data_table(
    const double value,
    const std::string value_string,
    const std::shared_ptr <dealii::TableHandler> data_table)
{
    data_table->add_value(value_string, value);
    data_table->set_precision(value_string, 16);
    data_table->set_scientific(value_string, true);
}

template <int dim, int nstate>
void DecayingHomogeneousIsotropicTurbulenceInitCheck<dim, nstate>::read_data_file(
    std::string data_table_filename,
    std::shared_ptr<DGBase<dim,double>> dg) const
{
    if(mpi_rank==0) {
        std::string line;
        // std::string::size_type sz1;

        std::ifstream FILE (data_table_filename);
        std::getline(FILE, line); // read first line: DOFs
        
        // check that the file is not empty
        if (line.empty()) {
            pcout << "Error: Trying to read empty file named " << data_table_filename << std::endl;
            std::abort();
        }

        // check that DOFs are correct
        const int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        const int expected_dofs = std::stoi(line);
        if(number_of_degrees_of_freedom_per_state != expected_dofs) {
            pcout << "Error: Number of degrees of freedom does not match the DHIT flow setup file: " << data_table_filename << std::endl;
            pcout << " -- Expected: " << expected_dofs << std::endl;
            pcout << " -- Current: " << number_of_degrees_of_freedom_per_state << std::endl;
            std::abort();
        }

        // this should still be checked when setting the initial condition

        // const int number_of_columns = 6;

        // std::getline(FILE, line); // read first line of data
        
        // // check that there indeed is data to be read
        // if (line.empty()) {
        //     pcout << "Error: Table has no data to be read" << std::endl;
        //     std::abort();
        // }

        // std::vector<double> current_line_values(number_of_columns);
        // while (!line.empty()) {
        //     std::string dummy_line = line;

        //     current_line_values[0] = std::stod(dummy_line,&sz1);
        //     for(int i=1; i<number_of_columns; ++i) {
        //         dummy_line = dummy_line.substr(sz1);
        //         sz1 = 0;
        //         current_line_values[i] = std::stod(dummy_line,&sz1);
        //     }

        //     // Add data entries to table
        //     for(int i=0; i<number_of_columns; ++i) {
        //         current_line_values[i];
                
        //     }
        //     std::getline(FILE, line); // read next line
        // }
    }
}

template <int dim, int nstate>
int DecayingHomogeneousIsotropicTurbulenceInitCheck<dim, nstate>::run_test() const
{
    // copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&parameters, parameter_handler);

    // read the setup.dat file; have a condition that checks that the DOFs are the same by reading the first line
    const std::string dhit_setup_file = "/home/julien/Codes/DHIT/example/setup.dat";
    read_data_file(dhit_setup_file,flow_solver->dg);
    // return 0;

    // write the reordering to a data table 

    if(this->mpi_rank==0) {
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();

        DGBase<dim, double> &dg = *(flow_solver->dg);

        // -- print all the points
        // // Overintegrate the error to make sure there is not integration error in the error estimate
        int overintegrate = 0;
        dealii::QGaussLobatto<dim> quad_extra(dg.max_degree+1+overintegrate);
        dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                                  dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

        const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
        // std::array<double,nstate> soln_at_q;
        // std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

        std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
        for (auto cell : dg.dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            fe_values_extra.reinit (cell);
            cell->get_dof_indices (dofs_indices);

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                // std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                // for (int s=0; s<nstate; ++s) {
                //     for (int d=0; d<dim; ++d) {
                //         soln_grad_at_q[s][d] = 0.0;
                //     }
                // }
                // for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                //     const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                //     soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                //     soln_grad_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_grad_component(idof,iquad,istate);
                // }
                const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                add_value_to_data_table(qpoint[0], "x", unsteady_data_table);
                add_value_to_data_table(qpoint[1], "y", unsteady_data_table);
                add_value_to_data_table(qpoint[2], "z", unsteady_data_table);
            }
        }
        const std::string file_write = "coordinates_check.txt";
        std::ofstream unsteady_data_table_file(file_write);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    // const double time_at_which_we_stop_the_run = 6.2831853072000017e-03;// chosen from running test MPI_VISCOUS_TAYLOR_GREEN_VORTEX_ENERGY_CHECK_QUICK
    // const int restart_file_index = 4;
    // const int initial_iteration_restart = restart_file_index; // assumes output mod for restart files is 1
    // const double time_at_which_the_run_is_complete = this->all_parameters->flow_solver_param.final_time;
    // Parameters::AllParameters params_incomplete_run = reinit_params(true,false,time_at_which_we_stop_the_run);

    // // Integrate to time at which we stop the run
    // std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_incomplete_run = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params_incomplete_run, parameter_handler);
    
    // static_cast<void>(flow_solver_incomplete_run->run());

    // const double time_step_at_stop_time = flow_solver_incomplete_run->flow_solver_case->get_constant_time_step(flow_solver_incomplete_run->dg);
    // const double desired_time_for_output_solution_every_dt_time_intervals_at_stop_time = flow_solver_incomplete_run->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals;
    // Parameters::AllParameters params_restart_to_complete_run = reinit_params(false,true,
    //                                                                          time_at_which_the_run_is_complete,
    //                                                                          time_at_which_we_stop_the_run,
    //                                                                          initial_iteration_restart,
    //                                                                          desired_time_for_output_solution_every_dt_time_intervals_at_stop_time,
    //                                                                          time_step_at_stop_time,
    //                                                                          restart_file_index);

    // // INLINE SUB-TEST: Check whether the initialize_data_table_from_file() function in flow solver is working correctly
    // if(this->mpi_rank==0) {
    //     std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
    //     const std::string file_read = params_incomplete_run.flow_solver_param.restart_files_directory_name+std::string("/")+params_incomplete_run.flow_solver_param.unsteady_data_table_filename+std::string("-")+flow_solver_incomplete_run->get_restart_filename_without_extension(restart_file_index)+std::string(".txt");
    //     flow_solver_incomplete_run->initialize_data_table_from_file(file_read,unsteady_data_table);
    //     const std::string file_write = "read_table_check.txt";
    //     std::ofstream unsteady_data_table_file(file_write);
    //     unsteady_data_table->write_text(unsteady_data_table_file);
    //     // check if files are the same (i.e. if the tables are the same)
    //     bool files_are_same = compare_files(file_read,file_write);
    //     if(!files_are_same) {
    //         pcout << "\n Error: initialize_data_table_from_file() failed." << std::endl;
    //         return 1;
    //     }
    //     else {
    //         pcout << "\n Sub-test for initialize_data_table_from_file() passed, continuing test..." << std::endl;
    //     }
    // } // END

    // // Integrate to final time by restarting from where we stopped
    // std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_restart_to_complete_run = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params_restart_to_complete_run, parameter_handler);
    // static_cast<void>(flow_solver_restart_to_complete_run->run());

    // // Compute kinetic energy at final time achieved by restarting the computation
    // std::unique_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicTurbulence<dim,nstate>>(this->all_parameters);
    // flow_solver_case->compute_and_update_integrated_quantities(*(flow_solver_restart_to_complete_run->dg));
    // const double kinetic_energy_computed = flow_solver_case->get_integrated_kinetic_energy();

    // const double relative_error = abs(kinetic_energy_computed - kinetic_energy_expected)/kinetic_energy_expected;
    // if (relative_error > 1.0e-10) {
    //     pcout << "Computed kinetic energy is not within specified tolerance with respect to expected kinetic energy." << std::endl;
    //     return 1;
    // }
    // pcout << " Test passed, computed kinetic energy is within specified tolerance." << std::endl;
    return 0;
}

#if PHILIP_DIM==3
    template class DecayingHomogeneousIsotropicTurbulenceInitCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace