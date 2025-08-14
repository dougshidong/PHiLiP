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

#include <boost/mpi/collectives.hpp>

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
, do_compute_unsteady_data_and_write_to_table(flow_solver_param.do_compute_unsteady_data_and_write_to_table)
, dg(DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, flow_solver_param.max_poly_degree_for_adaptation, grid_degree, flow_solver_case->generate_grid()))
{
    pcout << "IN FLOW_SOLVER." << std::endl;
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
        if(flow_solver_param.compute_time_averaged_solution && (ode_solver->current_time > flow_solver_param.time_to_start_averaging)) {
            dg->triangulation->load(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension + std::string("_time_averaged"));
            dealii::LinearAlgebra::distributed::Vector<double> time_averaged_solution_no_ghost;
            time_averaged_solution_no_ghost.reinit(dg->locally_owned_dofs, this->mpi_communicator);
            dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> time_averaged_solution_transfer(dg->dof_handler);
            time_averaged_solution_transfer.deserialize(time_averaged_solution_no_ghost);
            dg->time_averaged_solution = time_averaged_solution_no_ghost; //< assignment
        }
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
        std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
        system_matrix->copy_from(dg->system_matrix);
        // I do not like what I did above. I just copied the system matrix, stored it in a shared pointer, then pass it below.
        // This will double the memory requirement of the system_matrix...
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

    if(flow_solver_param.compute_time_averaged_solution && (ode_solver->current_time > flow_solver_param.time_to_start_averaging)) {
        // time-averaged solution files
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> time_averaged_solution_transfer(dg->dof_handler);
        time_averaged_solution_transfer.prepare_for_serialization(dg->time_averaged_solution);
        dg->triangulation->save(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension +  std::string("_time_averaged"));
    }
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
void FlowSolver<dim,nstate>::perk_partitioning() const
{
    this->dg->assemble_residual();
    //this->dg->cell_volume.update_ghost_values();

    const std::size_t n_groups = this->ode_solver->group_ID.size();
    const unsigned int n_cells = dg->triangulation->n_active_cells();

    this->locations_to_evaluate_rhs_1.resize(n_groups);
    this->locations_rhs_1.resize(n_groups);
    this->all_locations_rhs_1.resize(n_groups);

    for (std::size_t i = 0; i < n_groups; ++i) {
        this->locations_to_evaluate_rhs_1[i].reinit(dg->triangulation->n_active_cells());
        this->locations_to_evaluate_rhs_1[i] = 0;
    }
    for (std::size_t i = 0; i < n_groups; ++i) {
        this->locations_rhs_1[i].resize(dg->triangulation->n_active_cells());
    }

    //dg->cell_volume.print(std::cout);
    //std::cout << "number of cells " << dg->cell_volume.size() << std::endl;

    double local_max = dg->cell_volume.linfty_norm();
    double max_cell_volume = dealii::Utilities::MPI::max(local_max, this->mpi_communicator);
    //std::cout << "max cell volume " << max_cell_volume << " " << dg->cell_volume.size() << std::endl;
    //int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    for (typename dealii::DoFHandler<dim>::active_cell_iterator cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned())
            continue;
        double vol = dg->cell_volume[cell->active_cell_index()];
        for (std::size_t i = 0; i < n_groups; ++i) {
            if (vol >= 0.05 * max_cell_volume && i == 0) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            } else if (i == 1 && vol >= 0.005 * max_cell_volume && vol < 0.05 * max_cell_volume) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            } else if (i == 2 && vol >= 0.0001 * max_cell_volume && vol < 0.005 * max_cell_volume) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            } else if (i == 3 && vol >= 0.000005 * max_cell_volume && vol < 0.0001 * max_cell_volume) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            } else if (i == 4 && vol >= 0.0000006 * max_cell_volume && vol < 0.000005 * max_cell_volume) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            } else if (i == 5 && vol < 0.0000006 * max_cell_volume) {
                locations_to_evaluate_rhs_1[i](cell->active_cell_index()) = 1;
            }
        }
        //cell_weights[cell->active_cell_index()] = static_cast<unsigned int>(std::ceil(dg->cell_volume[cell->active_cell_index()] / max_cell_volume * 100));
    }

    std::vector<unsigned int> indices(n_cells);
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t i = 0; i < n_groups; ++i){
        for (std::size_t t = 0; t < n_cells; ++t){
            this->locations_rhs_1[i][t] = this->locations_to_evaluate_rhs_1[i][t];
        }
        this->all_locations_rhs_1[i] = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, this->locations_rhs_1[i]);

        const std::size_t n_ranks = this->all_locations_rhs_1[i].size();
        for (std::size_t idx = 0; idx < n_ranks; ++idx){
            //std::cout << this->mpi_rank << " " << idx << '\n';
            if (idx != static_cast<std::size_t>(this->mpi_rank))
                this->locations_to_evaluate_rhs_1[i].add(indices, this->all_locations_rhs_1[i][idx]);
        }
        this->locations_to_evaluate_rhs_1[i].compress(dealii::VectorOperation::insert);
        this->locations_to_evaluate_rhs_1[i].update_ghost_values();
        dg->set_list_of_cell_group_IDs(this->locations_to_evaluate_rhs_1[i], this->ode_solver->group_ID[i]);
    }

    // std::cout << "rhs 1 " << rank << std::endl;
    // this->locations_to_evaluate_rhs_1[0].print(std::cout);

    // std::cout << "rhs 2 " << rank << std::endl;
    // this->locations_to_evaluate_rhs_1[1].print(std::cout);

    //const unsigned int n_partitions = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    //std::vector<unsigned int> cell_weights(dg->triangulation->n_active_cells());

    // dealii::GridTools::partition_triangulation(
    //         n_partitions,         
    //         cell_weights,         
    //         *dg->triangulation,      
    //         dealii::SparsityTools::Partitioner::metis);




//    Partitioning
    // std::cout<< "partitioning" <<std::endl;
    //     locations_to_evaluate_rhs.reinit(dg->triangulation->n_active_cells());
    //     evaluate_until_this_index = locations_to_evaluate_rhs.size() / 2; 

    //     for (int i = 0; i < evaluate_until_this_index; ++i){
    //         if (locations_to_evaluate_rhs.in_local_range(i))
    //             locations_to_evaluate_rhs(i) = 1;
    //     }
    //     locations_to_evaluate_rhs.update_ghost_values();
    //     locations_to_evaluate_rhs.print(std::cout);

    //     dg->set_list_of_cell_group_IDs(locations_to_evaluate_rhs, this->ode_solver->group_ID[1]);


    //     locations_to_evaluate_rhs *= 0;
    //     locations_to_evaluate_rhs.update_ghost_values();

    //     for (size_t i = evaluate_until_this_index; i < locations_to_evaluate_rhs.size(); ++i){
    //         if (locations_to_evaluate_rhs.in_local_range(i))
    //             locations_to_evaluate_rhs(i) = 1;
    //     }
    //     locations_to_evaluate_rhs.update_ghost_values();
    //     locations_to_evaluate_rhs.print(std::cout);
    //     dg->set_list_of_cell_group_IDs(locations_to_evaluate_rhs, this->ode_solver->group_ID[0]); 
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
            ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_start_time + ode_param.output_solution_every_dt_time_intervals;
        } else if (this->do_output_solution_at_fixed_times && (this->number_of_fixed_times_to_output_solution > 0)) {
            pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
            dg->output_results_vtk(ode_solver->current_iteration);
        }
    }
    PHiLiP::Parameters::AllParameters parameters = *(dg->all_parameters);
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    if (parameters.ode_solver_param.ode_solver_type == ODESolverEnum::PERK_solver){
        perk_partitioning();
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
        double current_desired_time_for_output_restart_files_every_dt_time_intervals = ode_solver->current_time;
        unsigned int current_restart_file_number = 1;
        if(flow_solver_param.output_restart_files == true) {
            if(flow_solver_param.output_restart_files_every_dt_time_intervals > 0.0) {
                while(current_desired_time_for_output_restart_files_every_dt_time_intervals <= ode_solver->current_time) {
                    current_desired_time_for_output_restart_files_every_dt_time_intervals += flow_solver_param.output_restart_files_every_dt_time_intervals;
                }
            }
        }
        if(flow_solver_param.restart_computation_from_file == true) {
            current_restart_file_number = flow_solver_param.restart_file_index + 1;
        }
#endif
        //--------------------------------------------------------------------
        // Initialize the time at which we write the unsteady data table
        //--------------------------------------------------------------------
        double current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals = ode_solver->current_time;
        if(flow_solver_param.write_unsteady_data_table_file_every_dt_time_intervals > 0.0) {
            while(current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals <= ode_solver->current_time) {
                current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals += flow_solver_param.write_unsteady_data_table_file_every_dt_time_intervals;
            }
        }
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
        //dg->set_unsteady_model_time_step(time_step);
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
            if(do_compute_unsteady_data_and_write_to_table){
                pcout << "Writing unsteady data computed at initial time... " << std::endl;
                flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table, true);
                pcout << "done." << std::endl;
            }
        }
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        double next_time_step = time_step;
        std::shared_ptr<dealii::TableHandler> timer_values_table = std::make_shared<dealii::TableHandler>();
        pcout << "Advancing solution in time... " << std::endl;
        pcout << "Timer starting. " << std::endl;
        dealii::Timer timer(this->mpi_communicator,false);
        timer.start();
        while(ode_solver->current_time < final_time)
        {
            time_step = next_time_step; // update time step
            pcout<<"Iter: "<<ode_solver->current_iteration<<".  Current time: "<<ode_solver->current_time<<".\n";

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
            //dg->set_unsteady_model_time_step(time_step);
            // // // advance solution
            // // pcout<<"\n\nSet time step.\n";
            // const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
            // std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            // auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
            // for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            //     if (!current_cell->is_locally_owned()) continue;
            //     const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();
            //     const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
            //     current_dofs_indices.resize(n_dofs_cell);
            //     current_cell->get_dof_indices (current_dofs_indices);
            //     pcout<<"\n\ncurrent_cell_index: "<<current_cell_index;        
            //     for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            //         pcout<<"\ndg->solution(idof = "<<idof<<"): "<<dg->solution(current_dofs_indices[idof]);
            //     }
            // }
            // pcout<<"Step_in_time.\n";
            ode_solver->step_in_time(time_step,false); // pseudotime==false


            // pcout<<"Current time: "<<ode_solver->current_time<<"\n";
            // pcout<<"Time step: "<<time_step;
            // auto metric_cell_1 = dg->high_order_grid->dof_handler_grid.begin_active();
            // for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell_1) {
            //     if (!current_cell->is_locally_owned()) continue;
            //     const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();
            //     if(current_cell_index==1){
            //         const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
            //         current_dofs_indices.resize(n_dofs_cell);
            //         current_cell->get_dof_indices (current_dofs_indices);             
            //         for(unsigned int idof=0; idof<2; idof++){
            //             pcout<<"\nCell 1 solution (idof = "<<idof<<"): "<<dg->solution(current_dofs_indices[idof]);
            //         }
            //     }
            // }
            if constexpr (nstate==dim+2){
                if(flow_solver_param.compute_time_averaged_solution && ( (ode_solver->current_time <= flow_solver_param.time_to_start_averaging) && (ode_solver->current_time+time_step > flow_solver_param.time_to_start_averaging) )) {
                    dg->time_averaged_solution =  dg->solution;
                }
                else if(flow_solver_param.compute_time_averaged_solution && (ode_solver->current_time > flow_solver_param.time_to_start_averaging)) {
                    //dg->time_averaged_solution +=  dg->solution;
                    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
                    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
                    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
                    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                        if (!current_cell->is_locally_owned()) continue;

                        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
                        current_dofs_indices.resize(n_dofs_cell);
                        current_cell->get_dof_indices (current_dofs_indices);             
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            dg->time_averaged_solution(current_dofs_indices[idof]) = dg->time_averaged_solution(current_dofs_indices[idof]) + (dg->solution(current_dofs_indices[idof]) - dg->time_averaged_solution(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_averaging + time_step) / time_step); //Incremental average
                        }
                    }
                }
                if(flow_solver_param.compute_Reynolds_stress && ( (ode_solver->current_time <= flow_solver_param.time_to_start_computing_Reynolds_Stress) && (ode_solver->current_time+time_step > flow_solver_param.time_to_start_computing_Reynolds_Stress) )) {
                    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
                    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
                    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
                    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                        if (!current_cell->is_locally_owned()) continue;
                        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
                        current_dofs_indices.resize(n_dofs_cell);
                        current_cell->get_dof_indices (current_dofs_indices);
                        const unsigned int n_shape_fns = n_dofs_cell / nstate;
                        dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),poly_degree);
                        const unsigned int n_quad_pts = pow(vol_quad_equidistant_1D.size(),dim);
                        const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
                        OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, dg->max_degree, init_grid_degree); 
                        soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
                        soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);                
                        // Store solution coeffs for time-averaged flutuating quantitites
                        std::array<std::vector<double>,nstate> soln_coeff;
                        std::array<std::vector<double>,nstate> time_averaged_soln_coeff;
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                            if(ishape == 0) {
                                soln_coeff[istate].resize(n_shape_fns);
                                time_averaged_soln_coeff[istate].resize(n_shape_fns);
                            }
                            soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
                            time_averaged_soln_coeff[istate][ishape] = dg->time_averaged_solution(current_dofs_indices[idof]);
                        }

                        //Project solutin
                        std::array<std::vector<double>,nstate> soln_at_q;
                        std::array<std::vector<double>,nstate> time_averaged_soln_at_q;
                        for(int istate=0; istate<nstate; istate++){
                            soln_at_q[istate].resize(n_quad_pts);
                            time_averaged_soln_at_q[istate].resize(n_quad_pts);
                            // Interpolate soln coeff to volume cubature nodes.
                            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                                            soln_basis.oneD_vol_operator);
                            // Interpolate soln coeff to volume cubature nodes.
                            soln_basis.matrix_vector_mult_1D(time_averaged_soln_coeff[istate], time_averaged_soln_at_q[istate],
                                                            soln_basis.oneD_vol_operator);
                        }
                        // compute quantities at quad nodes (equisdistant)
                        dealii::Tensor<1,dim,std::vector<double>> velocity_at_q;
                        dealii::Tensor<1,dim,std::vector<double>> time_averaged_velocity_at_q;
                        dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
                        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                            std::array<double,nstate> soln_state;
                            std::array<double,nstate> time_averaged_soln_state;
                            for(int istate=0; istate<nstate; istate++){
                                soln_state[istate] = soln_at_q[istate][iquad];
                                time_averaged_soln_state[istate] = time_averaged_soln_at_q[istate][iquad];
                            }
                            dealii::Tensor<1,dim,double> vel;// = this->navier_stokes_physics->compute_velocities(soln_state);
                            dealii::Tensor<1,dim,double> time_averaged_vel;// = this->navier_stokes_physics->compute_velocities(time_averaged_soln_state);
                            const double density = soln_state[0];
                            const double time_averaged_density = time_averaged_soln_state[0];
                            for (unsigned int d=0; d<dim; ++d) {
                                vel[d] = soln_state[1+d]/density;
                                time_averaged_vel[d] = time_averaged_soln_state[1+d]/time_averaged_density;
                            }
                            const dealii::Tensor<1,dim,double> velocity = vel;
                            const dealii::Tensor<1,dim,double> time_averaged_velocity = time_averaged_vel;
                            for(int idim=0; idim<dim; idim++){
                                if(iquad==0){
                                    velocity_at_q[idim].resize(n_quad_pts);
                                    time_averaged_velocity_at_q[idim].resize(n_quad_pts);
                                    velocity_fluctuations_at_q[idim].resize(n_quad_pts);
                                }
                                velocity_at_q[idim][iquad] = velocity[idim];
                                time_averaged_velocity_at_q[idim][iquad] = time_averaged_velocity[idim];
                                velocity_fluctuations_at_q[idim][iquad] = velocity_at_q[idim][iquad] - time_averaged_velocity_at_q[idim][iquad];
                            } 
                        }
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                            if(istate == 0){//u^{\prime}v^{\prime}
                                dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[1][ishape];
                            }else if(istate == 1){//u^{\prime 2}
                                dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[0][ishape];
                            }else if(istate == 2){//v^{\prime 2}
                                dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[1][ishape]*velocity_fluctuations_at_q[1][ishape];
                            }else if(istate == 3){//z^{\prime 2}
                                dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[2][ishape]*velocity_fluctuations_at_q[2][ishape];
                            }else if(istate == 4){//u^{\prime}w^{\prime}
                                dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[2][ishape];
                            }
                        }
                    }
                }
                else if(flow_solver_param.compute_Reynolds_stress && (ode_solver->current_time > flow_solver_param.time_to_start_computing_Reynolds_Stress)) {
                    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
                    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
                    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
                    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                        if (!current_cell->is_locally_owned()) continue;

                        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
                        current_dofs_indices.resize(n_dofs_cell);
                        current_cell->get_dof_indices (current_dofs_indices);
                        const unsigned int n_shape_fns = n_dofs_cell / nstate;
                        dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),poly_degree);
                        const unsigned int n_quad_pts = pow(vol_quad_equidistant_1D.size(),dim);
                        const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
                        OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, dg->max_degree, init_grid_degree); 
                        soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
                        soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);                
                        // Store solution coeffs for time-averaged flutuating quantitites
                        std::array<std::vector<double>,nstate> soln_coeff;
                        std::array<std::vector<double>,nstate> time_averaged_soln_coeff;
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                            if(ishape == 0) {
                                soln_coeff[istate].resize(n_shape_fns);
                                time_averaged_soln_coeff[istate].resize(n_shape_fns);
                            }
                            soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
                            time_averaged_soln_coeff[istate][ishape] = dg->time_averaged_solution(current_dofs_indices[idof]);
                        }

                        //Project solutin
                        std::array<std::vector<double>,nstate> soln_at_q;
                        std::array<std::vector<double>,nstate> time_averaged_soln_at_q;
                        for(int istate=0; istate<nstate; istate++){
                            soln_at_q[istate].resize(n_quad_pts);
                            time_averaged_soln_at_q[istate].resize(n_quad_pts);
                            // Interpolate soln coeff to volume cubature nodes.
                            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                                            soln_basis.oneD_vol_operator);
                            // Interpolate soln coeff to volume cubature nodes.
                            soln_basis.matrix_vector_mult_1D(time_averaged_soln_coeff[istate], time_averaged_soln_at_q[istate],
                                                            soln_basis.oneD_vol_operator);
                        }
                        // compute quantities at quad nodes (equisdistant)
                        dealii::Tensor<1,dim,std::vector<double>> velocity_at_q;
                        dealii::Tensor<1,dim,std::vector<double>> time_averaged_velocity_at_q;
                        dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
                        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                            std::array<double,nstate> soln_state;
                            std::array<double,nstate> time_averaged_soln_state;
                            for(int istate=0; istate<nstate; istate++){
                                soln_state[istate] = soln_at_q[istate][iquad];
                                time_averaged_soln_state[istate] = time_averaged_soln_at_q[istate][iquad];
                            }
                            dealii::Tensor<1,dim,double> vel;// = this->navier_stokes_physics->compute_velocities(soln_state);
                            dealii::Tensor<1,dim,double> time_averaged_vel;// = this->navier_stokes_physics->compute_velocities(time_averaged_soln_state);
                            const double density = soln_state[0];
                            const double time_averaged_density = time_averaged_soln_state[0];
                            for (unsigned int d=0; d<dim; ++d) {
                                vel[d] = soln_state[1+d]/density;
                                time_averaged_vel[d] = time_averaged_soln_state[1+d]/time_averaged_density;
                            }
                            const dealii::Tensor<1,dim,double> velocity = vel;
                            const dealii::Tensor<1,dim,double> time_averaged_velocity = time_averaged_vel;
                            for(int idim=0; idim<dim; idim++){
                                if(iquad==0){
                                    velocity_at_q[idim].resize(n_quad_pts);
                                    time_averaged_velocity_at_q[idim].resize(n_quad_pts);
                                    velocity_fluctuations_at_q[idim].resize(n_quad_pts);
                                }
                                velocity_at_q[idim][iquad] = velocity[idim];
                                time_averaged_velocity_at_q[idim][iquad] = time_averaged_velocity[idim];
                                velocity_fluctuations_at_q[idim][iquad] = velocity_at_q[idim][iquad] - time_averaged_velocity_at_q[idim][iquad];
                            } 
                        }
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                            if(istate == 0){//u^{\prime}v^{\prime}
                                //dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[1][ishape];
                                dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[1][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_computing_Reynolds_Stress + time_step) / time_step); //Incremental average
                            }else if(istate == 1){//u^{\prime 2}
                                //dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[0][ishape];
                                dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[0][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_computing_Reynolds_Stress + time_step) / time_step); //Incremental average
                            }else if(istate == 2){//v^{\prime 2}
                                //dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[1][ishape]*velocity_fluctuations_at_q[1][ishape];
                                dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[1][ishape]*velocity_fluctuations_at_q[1][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_computing_Reynolds_Stress + time_step) / time_step); //Incremental average
                            }else if(istate == 3){//z^{\prime 2}
                                //dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[2][ishape]*velocity_fluctuations_at_q[2][ishape];
                                dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[2][ishape]*velocity_fluctuations_at_q[2][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_computing_Reynolds_Stress + time_step) / time_step); //Incremental average
                            }else if(istate == 4){//u^{\prime}w^{\prime}
                                //dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[2][ishape];
                                dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[2][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - flow_solver_param.time_to_start_computing_Reynolds_Stress + time_step) / time_step); //Incremental average
                            }
                        }
                    }
                }
            }
            bool do_write_unsteady_data_table_file = false;
            if(flow_solver_param.write_unsteady_data_table_file_every_dt_time_intervals > 0.0) {
                const bool is_write_time = ((ode_solver->current_time <= current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals) && 
                                             ((ode_solver->current_time + time_step) > current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals)) 
                                            || (ode_solver->current_time > current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals);
                if (is_write_time) {
                    do_write_unsteady_data_table_file = true;
                    current_desired_time_for_write_unsteady_data_table_file_every_dt_time_intervals += flow_solver_param.write_unsteady_data_table_file_every_dt_time_intervals;
                }
            } else {
                do_write_unsteady_data_table_file = true;
            }

            // Compute the unsteady quantities, write to the dealii table, and output to file
            if(do_compute_unsteady_data_and_write_to_table){
                flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table, do_write_unsteady_data_table_file);
            }
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
                                                 ((ode_solver->current_time + next_time_step) > current_desired_time_for_output_restart_files_every_dt_time_intervals)) 
                                                || (ode_solver->current_time > current_desired_time_for_output_restart_files_every_dt_time_intervals);
                    if (is_output_time) {
                        output_restart_files(current_restart_file_number, next_time_step, unsteady_data_table);
                        current_desired_time_for_output_restart_files_every_dt_time_intervals += flow_solver_param.output_restart_files_every_dt_time_intervals;
                        current_restart_file_number += 1;
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
                                             ((ode_solver->current_time + next_time_step) > ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals))
                                            || (ode_solver->current_time > ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals);
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
        const double cpu_time = timer.cpu_time();
        const double total_wall_time = dealii::Utilities::MPI::sum(timer.wall_time(), this->mpi_communicator);
        const double number_of_time_steps = (double)ode_solver->current_iteration;
        const double avg_cpu_time_per_time_step = cpu_time/number_of_time_steps;
        const double avg_total_wall_time_per_time_step = total_wall_time/number_of_time_steps;
        pcout << "Elapsed CPU time: " << cpu_time << " seconds." << std::endl;
        pcout << "Elapsed total wall time (mpi max): " << total_wall_time << " seconds." << std::endl;
        pcout << "Average CPU time per time step: " << avg_cpu_time_per_time_step << " seconds." << std::endl;
        pcout << "Average total wall time per time step: " << avg_total_wall_time_per_time_step << " seconds." << std::endl;
        // writing timing to file
        if(mpi_rank==0) {
            // add values to table
            flow_solver_case->add_value_to_data_table(cpu_time,"total_cpu_time",timer_values_table);
            flow_solver_case->add_value_to_data_table(total_wall_time,"total_wall_time",timer_values_table);
            flow_solver_case->add_value_to_data_table(avg_cpu_time_per_time_step,"avg_cpu_time",timer_values_table);
            flow_solver_case->add_value_to_data_table(avg_total_wall_time_per_time_step,"avg_wall_time",timer_values_table);
            std::string timing_table_filename = std::string("timer_values.txt");
            std::ofstream timer_values_table_file(timing_table_filename);
            timer_values_table->write_text(timer_values_table_file);
        }
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

