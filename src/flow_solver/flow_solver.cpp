#include "flow_solver.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "reduced_order/pod_basis_offline.h"
#include "reduced_order/pod_basis_online.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include <deal.II/base/timer.h>

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// FLOW SOLVER CLASS
//=========================================================
template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(
    const PHiLiP::Parameters::AllParameters *const parameters_input, 
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case_input,
    const dealii::ParameterHandler &parameter_handler_input)
: FlowSolverBase()
, flow_solver_case(flow_solver_case_input)
, parameter_handler(parameter_handler_input)
, mpi_communicator(MPI_COMM_WORLD)
, mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
, n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
, pcout(std::cout, mpi_rank==0)
, all_param(*parameters_input)
, flow_solver_param(all_param.flow_solver_param)
, ode_param(all_param.ode_solver_param)
, poly_degree(flow_solver_param.poly_degree)
, grid_degree(flow_solver_param.grid_degree)
, final_time(flow_solver_param.final_time)
, input_parameters_file_reference_copy_filename(flow_solver_param.restart_files_directory_name + std::string("/") + std::string("input_copy.prm"))
, do_output_solution_at_fixed_times(ode_param.output_solution_at_fixed_times)
, number_of_fixed_times_to_output_solution(ode_param.number_of_fixed_times_to_output_solution)
, output_solution_at_exact_fixed_times(ode_param.output_solution_at_exact_fixed_times)
, dg(DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, flow_solver_param.max_poly_degree_for_adaptation, grid_degree, flow_solver_case->generate_grid()))
{
    flow_solver_case->set_higher_order_grid(dg);
    if (ode_param.allocate_matrix_dRdW) {
        pcout << "Note: Allocating DG with AD matrix dRdW only." << std::endl;
        dg->allocate_system(true,false,false); // FlowSolver only requires dRdW to be allocated
    } else {
        pcout << "Note: Allocating DG without AD matrices." << std::endl;
        dg->allocate_system(false,false,false);
    }


    flow_solver_case->display_flow_solver_setup(dg);

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
        dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
        solution_no_ghost.reinit(dg->locally_owned_dofs, this->mpi_communicator);
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.deserialize(solution_no_ghost);
        dg->solution = solution_no_ghost; //< assignment
#endif
        pcout << "done." << std::endl;
    } else {
        // Initialize solution
        SetInitialCondition<dim,nstate,double>::set_initial_condition(flow_solver_case->initial_condition_function, dg, &all_param);
    }
    dg->solution.update_ghost_values();

    if(ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_solver || 
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_petrov_galerkin_solver ||
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_runge_kutta_solver){
        std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(dg);
        ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg, pod);
    } else {
        ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    }

    // Allocate ODE solver after initializing DG
    ode_solver->allocate_ode_system();

    // Storing a time_dependent POD
    const bool unsteady_FOM_POD_bool = all_param.reduced_order_param.output_snapshot_every_x_timesteps != 0 && !(ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_solver || 
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_petrov_galerkin_solver ||
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_runge_kutta_solver);
    if(unsteady_FOM_POD_bool){
        std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix(
            dg,
            &dg->system_matrix
        );
        time_pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix); 
        time_pod->addSnapshot(dg->solution);
    }

    // output a copy of the input parameters file
    if(flow_solver_param.output_restart_files == true) {
        pcout << "Writing a reference copy of the inputted parameters (.prm) file... " << std::flush;
        if(mpi_rank==0) {
            parameter_handler.print_parameters(input_parameters_file_reference_copy_filename);    
        }
        pcout << "done." << std::endl;
    }

    // For outputting solution at fixed times
    if(this->do_output_solution_at_fixed_times && (this->number_of_fixed_times_to_output_solution > 0)) {
        this->output_solution_fixed_times.reinit(this->number_of_fixed_times_to_output_solution);
        
        // Get output_solution_fixed_times from string
        const std::string output_solution_fixed_times_string = this->ode_param.output_solution_fixed_times_string;
        std::string line = output_solution_fixed_times_string;
        std::string::size_type sz1;
        this->output_solution_fixed_times[0] = std::stod(line,&sz1);
        for(unsigned int i=1; i<this->number_of_fixed_times_to_output_solution; ++i) {
            line = line.substr(sz1);
            sz1 = 0;
            this->output_solution_fixed_times[i] = std::stod(line,&sz1);
        }
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
std::string FlowSolver<dim,nstate>::get_restart_filename_without_extension(const unsigned int restart_index_input) const {
    // returns the restart file index as a string with appropriate padding
    std::string restart_index_string = std::to_string(restart_index_input);
    const unsigned int length_of_index_with_padding = 5;
    const unsigned int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
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
    if(mpi_rank==0) {
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
    const unsigned int restart_index_input,
    const double time_step_input) const {
    // write the restart parameter file
    if(mpi_rank==0) {
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
        ODE_solver_restart_parameter_values.push_back(double_to_string(time_step_input));


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
    const unsigned int current_restart_index,
    const double time_step_input,
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
    if(mpi_rank==0) {
        std::string restart_unsteady_data_table_filename = flow_solver_param.unsteady_data_table_filename+std::string("-")+restart_filename_without_extension+std::string(".txt");
        std::ofstream unsteady_data_table_file(flow_solver_param.restart_files_directory_name + std::string("/") + restart_unsteady_data_table_filename);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    // parameter file; written last to ensure necessary data/solution files have been written before
    write_restart_parameter_file(current_restart_index, time_step_input);
}
#endif

template <int dim, int nstate>
void FlowSolver<dim,nstate>::perform_steady_state_mesh_adaptation() const
{
    std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation = std::make_unique<MeshAdaptation<dim,double>>(this->dg, &(this->all_param.mesh_adaptation_param));
    const int total_adaptation_cycles = this->all_param.mesh_adaptation_param.total_mesh_adaptation_cycles;
    double residual_norm = this->dg->get_residual_l2norm();
    
    pcout<<"Running mesh adaptation cycles..."<<std::endl;
    while (meshadaptation->current_mesh_adaptation_cycle < total_adaptation_cycles)
    {
        // Check if steady state solution is being used.
        if(residual_norm > ode_param.nonlinear_steady_residual_tolerance)
        {
            pcout<<"Mesh adaptation is currently implemented for steady state flows and the current residual norm isn't sufficiently low. "
                 <<"The solution has not converged. If p or hp adaptation is being used, issues with convergence might occur when integrating face terms with lower quad points at " 
                 <<"the face of adjacent elements with different p. Try increasing overintegration in the parameters file to fix it."<<std::endl;
            std::abort();
        }
        
        meshadaptation->adapt_mesh();
        this->ode_solver->steady_state();
        residual_norm = this->ode_solver->residual_norm;
        flow_solver_case->steady_state_postprocessing(dg); 
    }

    pcout<<"Finished running mesh adaptation cycles."<<std::endl; 
}

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run() const
{
    pcout << "Running Flow Solver..." << std::endl;
    if(flow_solver_param.restart_computation_from_file == false) {
        if (ode_param.output_solution_every_x_steps > 0) {
            pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
            dg->output_results_vtk(ode_solver->current_iteration);
        } else if (ode_param.output_solution_every_dt_time_intervals > 0.0) {
            pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
            dg->output_results_vtk(ode_solver->current_iteration);
            ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
        } else if (this->do_output_solution_at_fixed_times && (this->number_of_fixed_times_to_output_solution > 0)) {
            pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
            dg->output_results_vtk(ode_solver->current_iteration);
        }
    }
    // Boolean to store solutions in POD object
    const bool unsteady_FOM_POD_bool = all_param.reduced_order_param.output_snapshot_every_x_timesteps != 0 && !(ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_solver || 
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_petrov_galerkin_solver ||
       ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_runge_kutta_solver);

    // Index of current desired fixed time to output solution
    unsigned int index_of_current_desired_fixed_time_to_output_solution = 0;
    
    // determine index_of_current_desired_fixed_time_to_output_solution if restarting solution
    if(flow_solver_param.restart_computation_from_file == true) {
        // use current_time to determine if restarting the computation from a non-zero initial time
        for(unsigned int i=0; i<this->number_of_fixed_times_to_output_solution; ++i) {
            if(this->ode_solver->current_time < this->output_solution_fixed_times[i]) {
                index_of_current_desired_fixed_time_to_output_solution = i;
                break;
            }
        }
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
        // Initialize time step
        //----------------------------------------------------
        double time_step = 0.0;
        if(flow_solver_param.adaptive_time_step == true && flow_solver_param.error_adaptive_time_step == true){
            pcout << "WARNING: CFL-adaptation and error-adaptation cannot be used at the same time. Aborting!" << std::endl;
            std::abort();
        }
        else if(flow_solver_param.adaptive_time_step == true) {
            pcout << "Setting initial adaptive time step... " << std::flush;
            time_step = flow_solver_case->get_adaptive_time_step_initial(dg);
        } else if(flow_solver_param.error_adaptive_time_step == true) {
            pcout << "Setting initial error adaptive time step... " << std::flush;
            time_step = ode_solver->get_automatic_initial_step_size(time_step,false);
        } else {
            pcout << "Setting constant time step... " << std::flush;
            time_step = flow_solver_case->get_constant_time_step(dg);
        }
        
        /* If restarting computation from file, it should give the same time step as written in file,
           a warning is thrown if this is not the case */
        if(flow_solver_param.restart_computation_from_file == true) {
            const double restart_time_step = ode_param.initial_time_step;
            if(std::abs(time_step-restart_time_step) > 1E-13) {
                pcout << "WARNING: Computed initial time step does not match value in restart parameter file within the tolerance. "
                      << "Diff is: " << std::abs(time_step-restart_time_step) << std::endl;
            }
        }
        flow_solver_case->set_time_step(time_step);
        pcout << "done." << std::endl;
        //----------------------------------------------------
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();
        if(flow_solver_param.restart_computation_from_file == true) {
            pcout << "Initializing data table from corresponding restart file... " << std::flush;
            const std::string restart_filename_without_extension = get_restart_filename_without_extension(flow_solver_param.restart_file_index);
            const std::string restart_unsteady_data_table_filename = flow_solver_param.unsteady_data_table_filename+std::string("-")+restart_filename_without_extension+std::string(".txt");
            initialize_data_table_from_file(flow_solver_param.restart_files_directory_name + std::string("/") + restart_unsteady_data_table_filename,unsteady_data_table);
            pcout << "done." << std::endl;
        } else {
            // no restart:
            pcout << "Writing unsteady data computed at initial time... " << std::endl;
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver, dg, unsteady_data_table);
            pcout << "done." << std::endl;
        }
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        double next_time_step = time_step;
        pcout << "Advancing solution in time... " << std::endl;
        pcout << "Timer starting. " << std::endl;
        dealii::Timer timer(this->mpi_communicator,false);
        timer.start();
        while(ode_solver->current_time < final_time)
        {
            time_step = next_time_step; // update time step

            // check if we need to decrease the time step
            if((ode_solver->current_time+time_step) > final_time && flow_solver_param.end_exactly_at_final_time) {
                // decrease time step to finish exactly at specified final time
                time_step = final_time - ode_solver->current_time;
            } else if (this->output_solution_at_exact_fixed_times && (this->do_output_solution_at_fixed_times && (this->number_of_fixed_times_to_output_solution > 0))) { // change this to some parameter
                const double next_time = ode_solver->current_time + time_step;
                const double desired_time = this->output_solution_fixed_times[index_of_current_desired_fixed_time_to_output_solution];
                // Check if current time is an output time
                const bool is_output_time = ((ode_solver->current_time<desired_time) && (next_time>desired_time));
                if(is_output_time) time_step = desired_time - ode_solver->current_time;
            }

            // update time step in flow_solver_case
            flow_solver_case->set_time_step(time_step);

            ode_solver->step_in_time(time_step,false);

            // Compute the unsteady quantities, write to the dealii table, and output to file
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver, dg, unsteady_data_table);
            // update next time step
                       
            if(flow_solver_param.adaptive_time_step == true) {
                next_time_step = flow_solver_case->get_adaptive_time_step(dg);
            } else if (flow_solver_param.error_adaptive_time_step == true) {
                next_time_step = ode_solver->get_automatic_error_adaptive_step_size(time_step,false); 
            } else {
                next_time_step = flow_solver_case->get_constant_time_step(dg);
            }
                      
            

#if PHILIP_DIM>1
            if(flow_solver_param.output_restart_files == true) {
                // Output restart files
                if(flow_solver_param.output_restart_files_every_dt_time_intervals > 0.0) {
                    const bool is_output_time = ((ode_solver->current_time <= current_desired_time_for_output_restart_files_every_dt_time_intervals) && 
                                                 ((ode_solver->current_time + next_time_step) > current_desired_time_for_output_restart_files_every_dt_time_intervals));
                    if (is_output_time) {
                        const unsigned int file_number = int(round(current_desired_time_for_output_restart_files_every_dt_time_intervals / flow_solver_param.output_restart_files_every_dt_time_intervals));
                        output_restart_files(file_number, next_time_step, unsteady_data_table);
                        current_desired_time_for_output_restart_files_every_dt_time_intervals += flow_solver_param.output_restart_files_every_dt_time_intervals;
                    }
                } else /*if (flow_solver_param.output_restart_files_every_x_steps > 0)*/ {
                    const bool is_output_iteration = (ode_solver->current_iteration % flow_solver_param.output_restart_files_every_x_steps == 0);
                    if (is_output_iteration) {
                        const unsigned int file_number = ode_solver->current_iteration / flow_solver_param.output_restart_files_every_x_steps;
                        output_restart_files(file_number, next_time_step, unsteady_data_table);
                    }
                }
            }
#endif

            // Output vtk solution files for post-processing in Paraview
            if (ode_param.output_solution_every_x_steps > 0) {
                const bool is_output_iteration = (ode_solver->current_iteration % ode_param.output_solution_every_x_steps == 0);
                if (is_output_iteration) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const unsigned int file_number = ode_solver->current_iteration / ode_param.output_solution_every_x_steps;
                    dg->output_results_vtk(file_number,ode_solver->current_time);
                }
            } else if(ode_param.output_solution_every_dt_time_intervals > 0.0) {
                const bool is_output_time = ((ode_solver->current_time <= ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals) && 
                                             ((ode_solver->current_time + next_time_step) > ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals));
                if (is_output_time) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const unsigned int file_number = int(round(ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals / ode_param.output_solution_every_dt_time_intervals));
                    dg->output_results_vtk(file_number,ode_solver->current_time);
                    ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
                }
            } else if (this->do_output_solution_at_fixed_times && (this->number_of_fixed_times_to_output_solution > 0)) {
                const double next_time = ode_solver->current_time + next_time_step;
                const double desired_time = this->output_solution_fixed_times[index_of_current_desired_fixed_time_to_output_solution];
                // Check if current time is an output time
                bool is_output_time = false; // default initialization
                if(this->output_solution_at_exact_fixed_times) {
                    is_output_time = ode_solver->current_time == desired_time;
                } else {
                    is_output_time = ((ode_solver->current_time<=desired_time) && (next_time>desired_time));
                }
                if(is_output_time) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const int file_number = index_of_current_desired_fixed_time_to_output_solution+1; // +1 because initial time is 0
                    dg->output_results_vtk(file_number,ode_solver->current_time);
                    
                    // Update index s.t. it never goes out of bounds
                    if(index_of_current_desired_fixed_time_to_output_solution 
                        < (this->number_of_fixed_times_to_output_solution-1)) {
                        index_of_current_desired_fixed_time_to_output_solution += 1;
                    }
                }
            }
            // Add snapshots to snapshot matrix
            if(unsteady_FOM_POD_bool){
                const bool is_snapshot_iteration = (ode_solver->current_iteration % all_param.reduced_order_param.output_snapshot_every_x_timesteps == 0);
                if(is_snapshot_iteration) time_pod->addSnapshot(dg->solution);
            }
        } // close while

        // Print POD Snapshots to file
        if(unsteady_FOM_POD_bool){
            std::ofstream snapshot_file("solution_snapshots_iteration_" + std::to_string(ode_solver->current_iteration) + ".txt"); // Change ode_solver->current_iteration to size of matrix
            unsigned int precision = 16;
            time_pod->dealiiSnapshotMatrix.print_formatted(snapshot_file, precision, true, 0, "0"); 
            snapshot_file.close();
        }

        timer.stop();
        pcout << "Timer stopped. " << std::endl;
        const double max_wall_time = dealii::Utilities::MPI::max(timer.wall_time(), this->mpi_communicator);
        pcout << "Elapsed wall time (mpi max): " << max_wall_time << " seconds." << std::endl;
        pcout << "Elapsed CPU time: " << timer.cpu_time() << " seconds." << std::endl;
    } else {
        //----------------------------------------------------
        // Steady-state solution
        //----------------------------------------------------
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        if(flow_solver_param.steady_state_polynomial_ramping && (ode_param.ode_solver_type != ODEEnum::pod_galerkin_solver && ode_param.ode_solver_type != ODEEnum::pod_petrov_galerkin_solver && ode_param.ode_solver_type != ODEEnum::hyper_reduced_petrov_galerkin_solver)) {
            ode_solver->initialize_steady_polynomial_ramping(poly_degree);
        }

        ode_solver->steady_state();
        flow_solver_case->steady_state_postprocessing(dg);
        
        const bool use_isotropic_mesh_adaptation = (all_param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0) 
                                        && (all_param.mesh_adaptation_param.mesh_adaptation_type != Parameters::MeshAdaptationParam::MeshAdaptationType::anisotropic_adaptation);
        
        if(use_isotropic_mesh_adaptation)
        {
            perform_steady_state_mesh_adaptation();
        }
    }
    pcout << "done." << std::endl;
    return 0;
}

#if PHILIP_DIM==1
template class FlowSolver <PHILIP_DIM,PHILIP_DIM>;
template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

#if PHILIP_DIM!=1
template class FlowSolver <PHILIP_DIM,1>;
template class FlowSolver <PHILIP_DIM,2>;
template class FlowSolver <PHILIP_DIM,3>;
template class FlowSolver <PHILIP_DIM,4>;
template class FlowSolver <PHILIP_DIM,5>;
template class FlowSolver <PHILIP_DIM,6>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

