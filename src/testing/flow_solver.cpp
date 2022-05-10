#include "flow_solver.h"

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>

namespace PHiLiP {

namespace Tests {
//=========================================================
// FLOW SOLVER TEST CASE -- What runs the test
//=========================================================
template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(
    const PHiLiP::Parameters::AllParameters *const parameters_input, 
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case,
    const dealii::ParameterHandler &parameter_handler_input)
: TestsBase::TestsBase(parameters_input)
, flow_solver_case(flow_solver_case)
, parameter_handler(parameter_handler_input)
, all_param(*parameters_input)
, flow_solver_param(all_param.flow_solver_param)
, ode_param(all_param.ode_solver_param)
, poly_degree(all_param.grid_refinement_study_param.poly_degree)
, final_time(flow_solver_param.final_time)
, input_parameters_file_reference_copy_filename(flow_solver_param.restart_files_directory_name + std::string("/") + std::string("input_copy.prm"))
, dg(DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, flow_solver_case->generate_grid()))
, ode_solver(ODE::ODESolverFactory<dim, double>::create_ODESolver(dg))
{
    flow_solver_case->set_higher_order_grid(dg);
    dg->allocate_system();
    flow_solver_case->display_flow_solver_setup();
    
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);

    if(flow_solver_param.restart_computation_from_file == true) {
        if(dim == 1) {
            pcout << "Error: restart_computation_from_file is not possible for 1D. Set to false." << std::endl;
            std::abort();
        }

        if (flow_solver_param.steady_state == true) {
            pcout << "Error: Restart capability has not been fully implemented / tested for steady state computations." << std::endl;
            std::abort();
        }

        // Initialize solution from restart file
        pcout << "Initializing solution from restart file..." << std::flush;
        const std::string restart_filename_without_extension = get_restart_filename_without_extension(flow_solver_param.restart_file_index);
#if PHILIP_DIM>1
        dg->triangulation->load(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
        
        // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
        // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.deserialize(solution_no_ghost);
#endif
    } else {
        // Initialize solution from initial_condition_function
        pcout << "Initializing solution with initial condition function... " << std::flush;
        dealii::VectorTools::interpolate(dg->dof_handler, *(flow_solver_case->initial_condition_function), solution_no_ghost);
    }
    dg->solution = solution_no_ghost; //< assignment
    dg->solution.update_ghost_values();
    pcout << "done." << std::endl;
    ode_solver->allocate_ode_system();

    // output a copy of the input parameters file
    if(flow_solver_param.output_restart_files == true) {
        pcout << "Writing a reference copy of the inputted parameters (.prm) file... " << std::flush;
        if(this->mpi_rank==0) {
            parameter_handler.print_parameters(input_parameters_file_reference_copy_filename);    
        }
        pcout << "done." << std::endl;
    }
}

template <int dim, int nstate>
std::vector<std::string> FlowSolver<dim,nstate>::get_data_table_column_names(const std::string string_input) const
{
    /* returns the column names of a dealii::TableHandler object
       given the first line of the file */
    
    // Create object of istringstream and initialize assign input string
    std::istringstream iss(string_input);
    std::string word;

    // extract each name (no spaces)
    std::vector<std::string> names;
    while(iss >> word) {
        names.push_back(word.c_str());
    }
    return names;
}

template <int dim, int nstate>
std::string FlowSolver<dim,nstate>::get_restart_filename_without_extension(const int restart_index_input) const {
    // returns the restart file index as a string with appropriate padding
    std::string restart_index_string = std::to_string(restart_index_input);
    const unsigned int length_of_index_with_padding = 5;
    const int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
    restart_index_string.insert(0, number_of_zeros, '0');

    const std::string prefix = "restart-";
    const std::string restart_filename_without_extension = prefix+restart_index_string;

    return restart_filename_without_extension;
}

template <int dim, int nstate>
void FlowSolver<dim,nstate>::initialize_data_table_from_file(
    std::string data_table_filename,
    const std::shared_ptr <dealii::TableHandler> data_table) const
{
    if(this->mpi_rank==0) {
        std::string line;
        std::string::size_type sz1;

        std::ifstream FILE (data_table_filename);
        std::getline(FILE, line); // read first line: column headers
        
        // check that the file is not empty
        if (line.empty()) {
            pcout << "Error: Trying to read empty file named " << data_table_filename << std::endl;
            std::abort();
        }

        const std::vector<std::string> data_column_names = get_data_table_column_names(line);
        const int number_of_columns = data_column_names.size();

        std::getline(FILE, line); // read first line of data
        
        // check that there indeed is data to be read
        if (line.empty()) {
            pcout << "Error: Table has no data to be read" << std::endl;
            std::abort();
        }

        std::vector<double> current_line_values(number_of_columns);
        while (!line.empty()) {
            std::string dummy_line = line;

            current_line_values[0] = std::stod(dummy_line,&sz1);
            for(int i=1; i<number_of_columns; ++i) {
                dummy_line = dummy_line.substr(sz1);
                sz1 = 0;
                current_line_values[i] = std::stod(dummy_line,&sz1);
            }

            // Add data entries to table
            for(int i=0; i<number_of_columns; ++i) {
                data_table->add_value(data_column_names[i], current_line_values[i]);
                data_table->set_precision(data_column_names[i], 16);
                data_table->set_scientific(data_column_names[i], true);
            }
            std::getline(FILE, line); // read next line
        }
    }
}

template <int dim, int nstate>
std::string FlowSolver<dim,nstate>::double_to_string(const double value_input) const {
    // converts a double to a string with full precision
    std::stringstream ss;
    ss << std::scientific << std::setprecision(16) << value_input;
    std::string double_to_string = ss.str();
    return double_to_string;
}

template <int dim, int nstate>
void FlowSolver<dim,nstate>::write_restart_parameter_file(
    const int restart_index_input,
    const double constant_time_step_input) const {
    // write the restart parameter file
    if(this->mpi_rank==0) {
        // read a copy of the current parameters file
        std::ifstream CURRENT_FILE(input_parameters_file_reference_copy_filename);
        
        // create write file with appropriate postfix given the restart index input
        const std::string restart_filename = get_restart_filename_without_extension(restart_index_input)+std::string(".prm");
        std::ofstream RESTART_FILE(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename);

        // Lines to identify the subsections in the .prm file
        /* WARNING: (2) These must be in the order they appear in the .prm file
         */
        std::vector<std::string> subsection_line;
        subsection_line.push_back("subsection ODE solver");
        subsection_line.push_back("subsection flow_solver");
        // Number of subsections to change values in
        int number_of_subsections = subsection_line.size();


        /* WARNING: (1) Must put a space before and after each parameter string as done below
         *          (2) These must be in the order they appear in the .prm file
         */
        // -- names
        std::vector<std::string> ODE_solver_restart_parameter_names;
        ODE_solver_restart_parameter_names.push_back(" initial_desired_time_for_output_solution_every_dt_time_intervals ");
        ODE_solver_restart_parameter_names.push_back(" initial_iteration ");
        ODE_solver_restart_parameter_names.push_back(" initial_time ");
        ODE_solver_restart_parameter_names.push_back(" initial_time_step ");
        // -- corresponding values
        std::vector<std::string> ODE_solver_restart_parameter_values;
        ODE_solver_restart_parameter_values.push_back(double_to_string(ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals));
        ODE_solver_restart_parameter_values.push_back(std::to_string(ode_solver->current_iteration));
        ODE_solver_restart_parameter_values.push_back(double_to_string(ode_solver->current_time));
        ODE_solver_restart_parameter_values.push_back(double_to_string(constant_time_step_input));


        /* WARNING: (1) Must put a space before and after each parameter string as done below
         *          (2) These must be in the order they appear in the .prm file
         */
        // -- Names
        std::vector<std::string> flow_solver_restart_parameter_names;
        flow_solver_restart_parameter_names.push_back(" output_restart_files ");
        flow_solver_restart_parameter_names.push_back(" restart_computation_from_file ");
        flow_solver_restart_parameter_names.push_back(" restart_file_index ");
        // -- Corresponding values
        std::vector<std::string> flow_solver_restart_parameter_values;
        flow_solver_restart_parameter_values.push_back(std::string("true"));
        flow_solver_restart_parameter_values.push_back(std::string("true"));
        flow_solver_restart_parameter_values.push_back(std::to_string(restart_index_input));


        // Number of parameters in each subsection
        std::vector<int> number_of_subsection_parameters;
        number_of_subsection_parameters.push_back(ODE_solver_restart_parameter_names.size());
        number_of_subsection_parameters.push_back(flow_solver_restart_parameter_names.size());
        
        // Initialize for the while loop
        int i_subsection = 0;
        std::string line;

        // read line until end of file
        while (std::getline(CURRENT_FILE, line)) {
            // check if the desired subsection has been reached
            if (line == subsection_line[i_subsection]) {
                RESTART_FILE << line << "\n"; // write line

                int i_parameter = 0;
                std::string name;
                std::string value_string;

                if (i_subsection==0) {
                    name = ODE_solver_restart_parameter_names[i_parameter];
                    value_string = ODE_solver_restart_parameter_values[i_parameter];
                } else if (i_subsection==1) {
                    name = flow_solver_restart_parameter_names[i_parameter];
                    value_string = flow_solver_restart_parameter_values[i_parameter];
                }

                while (line!="end") {
                    std::getline(CURRENT_FILE, line); // read line
                    std::string::size_type found = line.find(name);
                    
                    // found the line corresponding to the desired parameter
                    if (found!=std::string::npos) {

                        // construct the updated line
                        std::string updated_line = line;
                        std::string::size_type position_to_replace = line.find_last_of("=")+2;
                        std::string part_of_line_to_replace = line.substr(position_to_replace);
                        updated_line.replace(position_to_replace,part_of_line_to_replace.length(),value_string);

                        // write updated line to restart file
                        RESTART_FILE << updated_line << "\n";
                        
                        // update the parameter index, name, and value
                        if ((i_parameter+1) < number_of_subsection_parameters[i_subsection]) ++i_parameter; // to avoid going out of bounds
                        if (i_subsection==0) {
                            name = ODE_solver_restart_parameter_names[i_parameter];
                            value_string = ODE_solver_restart_parameter_values[i_parameter];
                        } else if (i_subsection==1) {
                            name = flow_solver_restart_parameter_names[i_parameter];
                            value_string = flow_solver_restart_parameter_values[i_parameter];
                        }
                    } else {
                        // write line (that does correspond to the desired parameter) to the restart file
                        RESTART_FILE << line << "\n";
                    }
                }
                // update the subsection index
                if ((i_subsection+1) < number_of_subsections) ++i_subsection; // to avoid going out of bounds
            } else {
                // write line (that is not in a desired subsection) to the restart file
                RESTART_FILE << line << "\n";
            }
        }
    }
}

#if PHILIP_DIM>1
template <int dim, int nstate>
void FlowSolver<dim,nstate>::output_restart_files(
    const int current_restart_index,
    const double constant_time_step,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{
    pcout << "  ... Writing restart files ... " << std::endl;
    const std::string restart_filename_without_extension = get_restart_filename_without_extension(current_restart_index);

    // solution files
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
    // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    solution_transfer.prepare_for_serialization(dg->solution);
    dg->triangulation->save(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
    
    // unsteady data table
    if(this->mpi_rank==0) {
        std::string restart_unsteady_data_table_filename = flow_solver_param.unsteady_data_table_filename+std::string("-")+restart_filename_without_extension+std::string(".txt");
        std::ofstream unsteady_data_table_file(flow_solver_param.restart_files_directory_name + std::string("/") + restart_unsteady_data_table_filename);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    // parameter file; written last to ensure necessary data/solution files have been written before
    write_restart_parameter_file(current_restart_index, constant_time_step);
}
#endif

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run_test() const
{
    pcout << "Running Flow Solver..." << std::endl;
    if (ode_param.output_solution_every_x_steps > 0) {
        dg->output_results_vtk(ode_solver->current_iteration);
    } else if (ode_param.output_solution_every_dt_time_intervals > 0.0) {
        dg->output_results_vtk(ode_solver->current_iteration);
        ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
    }

    //----------------------------------------------------
    // Select unsteady or steady-state
    //----------------------------------------------------
    if(flow_solver_param.steady_state == false){
        //----------------------------------------------------
        //                  UNSTEADY FLOW
        //----------------------------------------------------
        // Initializing restart related variables
        //----------------------------------------------------
#if PHILIP_DIM>1
        double current_desired_time_for_output_restart_files_every_dt_time_intervals = ode_solver->current_time; // when used, same as the initial time
#endif
        //----------------------------------------------------
        // Constant time step based on CFL number
        //----------------------------------------------------
        pcout << "Setting constant time step... " << std::flush;
        const double constant_time_step = flow_solver_case->get_constant_time_step(dg);
        pcout << "done." << std::endl;
        //----------------------------------------------------
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
        if(flow_solver_param.restart_computation_from_file == true) {
            pcout << "Initializing data table from corresponding restart file... " << std::flush;
            const std::string restart_filename_without_extension = get_restart_filename_without_extension(flow_solver_param.restart_file_index);
            const std::string restart_unsteady_data_table_filename = flow_solver_param.unsteady_data_table_filename+std::string("-")+restart_filename_without_extension+std::string(".txt");
            initialize_data_table_from_file(flow_solver_param.restart_files_directory_name + std::string("/") + restart_unsteady_data_table_filename,unsteady_data_table);
            pcout << "done." << std::endl;
        } else {
            // no restart:
            pcout << "Writing unsteady data computed at initial time... " << std::endl;
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);
            pcout << "done." << std::endl;
        }
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        pcout << "Advancing solution in time... " << std::endl;
        while(ode_solver->current_time < final_time)
        {
            // advance solution
            ode_solver->step_in_time(constant_time_step,false); // pseudotime==false

            // Compute the unsteady quantities, write to the dealii table, and output to file
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);

#if PHILIP_DIM>1
            if(flow_solver_param.output_restart_files == true) {
                // Output restart files
                if(flow_solver_param.output_restart_files_every_dt_time_intervals > 0.0) {
                    const bool is_output_time = ((ode_solver->current_time <= current_desired_time_for_output_restart_files_every_dt_time_intervals) && 
                                                ((ode_solver->current_time + constant_time_step) > current_desired_time_for_output_restart_files_every_dt_time_intervals));
                    if (is_output_time) {
                        const int file_number = current_desired_time_for_output_restart_files_every_dt_time_intervals / flow_solver_param.output_restart_files_every_dt_time_intervals;
                        output_restart_files(file_number, constant_time_step, unsteady_data_table);
                        current_desired_time_for_output_restart_files_every_dt_time_intervals += flow_solver_param.output_restart_files_every_dt_time_intervals;
                    }
                } else /*if (flow_solver_param.output_restart_files_every_x_steps > 0)*/ {
                    const bool is_output_iteration = (ode_solver->current_iteration % flow_solver_param.output_restart_files_every_x_steps == 0);
                    if (is_output_iteration) {
                        const int file_number = ode_solver->current_iteration / flow_solver_param.output_restart_files_every_x_steps;
                        output_restart_files(file_number, constant_time_step, unsteady_data_table);
                    }
                }
            }
#endif

            // Output vtk solution files for post-processing in Paraview
            if (ode_param.output_solution_every_x_steps > 0) {
                const bool is_output_iteration = (ode_solver->current_iteration % ode_param.output_solution_every_x_steps == 0);
                if (is_output_iteration) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const int file_number = ode_solver->current_iteration / ode_param.output_solution_every_x_steps;
                    dg->output_results_vtk(file_number);
                }
            } else if(ode_param.output_solution_every_dt_time_intervals > 0.0) {
                const bool is_output_time = ((ode_solver->current_time <= ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals) && 
                                            ((ode_solver->current_time + constant_time_step) > ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals));
                if (is_output_time) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const int file_number = ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals / ode_param.output_solution_every_dt_time_intervals;
                    dg->output_results_vtk(file_number);
                    ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
                }
            }
        } // close while
    } else {
        //----------------------------------------------------
        // Steady-state solution
        //----------------------------------------------------
        ode_solver->steady_state();
        flow_solver_case->steady_state_postprocessing(dg);
    }
    pcout << "done." << std::endl;
    return 0;
}

//=========================================================
//                  FLOW SOLVER FACTORY
//=========================================================
template <int dim, int nstate>
std::unique_ptr < FlowSolver<dim,nstate> >
FlowSolverFactory<dim,nstate>
::create_FlowSolver(const Parameters::AllParameters *const parameters_input,
                    const dealii::ParameterHandler &parameter_handler_input)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex){
        if constexpr (dim==3 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicTurbulence<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot){
        if constexpr (dim==1 && nstate==dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersViscousSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersRewienskiSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::naca0012){
        if constexpr (dim==2 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NACA0012<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else {
        std::cout << "Invalid flow case. You probably forgot to add it to the list of flow cases in flow_solver.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

#if PHILIP_DIM==1
template class FlowSolver <PHILIP_DIM,PHILIP_DIM>;
template class FlowSolverFactory <PHILIP_DIM,PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
template class FlowSolverFactory <PHILIP_DIM,PHILIP_DIM+2>;
#endif


} // Tests namespace
} // PHiLiP namespace

