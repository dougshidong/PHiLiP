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
FlowSolver<dim, nstate>::FlowSolver(const PHiLiP::Parameters::AllParameters *const parameters_input, std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case)
: TestsBase::TestsBase(parameters_input)
, flow_solver_case(flow_solver_case)
, initial_condition_function(InitialConditionFactory<dim,double>::create_InitialConditionFunction(parameters_input, nstate))
, all_param(*parameters_input)
, flow_solver_param(all_param.flow_solver_param)
, ode_param(all_param.ode_solver_param)
, poly_degree(all_param.grid_refinement_study_param.poly_degree)
, final_time(flow_solver_param.final_time)
, dg(DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, flow_solver_case->generate_grid()))
, ode_solver(ODE::ODESolverFactory<dim, double>::create_ODESolver(dg))
{
    dg->allocate_system();
    flow_solver_case->display_flow_solver_setup();
    
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);

    if(flow_solver_param.restart_computation_from_file == true) {
        if(dim == 1) {
            pcout << "Error: restart_computation_from_file is not possible for 1D. Set to false." << std::endl;
            std::abort();
        }
#if PHILIP_DIM>1
        // Initialize solution from restart file
        pcout << "Initializing solution from restart file..." << std::flush;
        dg->triangulation->load("restart");
        // --- after allocate_dg
        // TO DO: Read section "Note on usage with DoFHandler with hp-capabilities" and add the stuff im missing
        // ------ Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        // dg->solution.zero_out_ghosts();
        solution_transfer.deserialize(solution_no_ghost);
        // dg->solution.update_ghost_values(); // -- should this be called when we do dg->solution = solution_no_ghost ?
        // if (flow_solver_param.steady_state == true) {
        //     pcout << "Error: Cannot use" << std::endl;
        //     std::abort();
        // }
#endif
    } else {
        // Initialize solution from initial_condition_function
        pcout << "Initializing solution with initial condition function..." << std::flush;
        dealii::VectorTools::interpolate(dg->dof_handler, *initial_condition_function, solution_no_ghost);
    }
    dg->solution = solution_no_ghost; //< assignment
    pcout << "done." << std::endl;
    ode_solver->allocate_ode_system();
}

template <int dim, int nstate>
std::vector<std::string> FlowSolver<dim,nstate>::get_data_table_column_names(const std::string string_input) const
{
    /* returns the column names of a dealii::TableHandler object
       given the first line of the file */
    
    // Crete object of istringstream and initialize assign input string
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
void FlowSolver<dim,nstate>::initialize_data_table_from_file(
    std::string data_table_filename_with_extension,
    const std::shared_ptr <dealii::TableHandler> data_table) const
{
    if(this->mpi_rank==0) {
        std::string line;
        std::string::size_type sz1;

        std::ifstream FILE (data_table_filename_with_extension);
        std::getline(FILE, line); // read first line: column headers
        
        // check that the file is not empty
        if (line.empty()) {
            pcout << "Error: Trying to read empty file" << std::endl;
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
            initialize_data_table_from_file(flow_solver_case->unsteady_data_table_filename_with_extension,unsteady_data_table);
            pcout << "done." << std::endl;
        } else {
            // no restart:
            pcout << "Writing unsteady data computed at initial time... " << std::flush;
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);
            pcout << "done." << std::endl;
        }
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        pcout << "Advancing solution in time... " << std::endl;
        while(ode_solver->current_time < final_time)
        {
            ode_solver->step_in_time(constant_time_step,false); // pseudotime==false

#if PHILIP_DIM>1
            // output the restart file
            if(flow_solver_param.output_restart_files == true) {
                pcout << "  ... Writing restart file ..." << std::endl;
                dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
                solution_transfer.prepare_for_coarsening_and_refinement(dg->solution);
                dg->triangulation->save("restart");
            }
#endif
            
            // Compute the unsteady quantities, write to the dealii table, and output to file
            flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);

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

template class FlowSolver <PHILIP_DIM,PHILIP_DIM>;
template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;

//=========================================================
// FLOW SOLVER FACTORY
//=========================================================
template <int dim, int nstate>
std::unique_ptr < FlowSolver<dim,nstate> >
FlowSolverFactory<dim,nstate>
::create_FlowSolver(const Parameters::AllParameters *const parameters_input)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex){
        if constexpr (dim==3 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case);
        }
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot){
        if constexpr (dim==1 && nstate==dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersViscousSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case);
        }
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersRewienskiSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case);
        }
    } else {
        std::cout << "Invalid flow case. You probably forgot to add it to the list of flow cases in flow_solver.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template class FlowSolverFactory <PHILIP_DIM,PHILIP_DIM>;
template class FlowSolverFactory <PHILIP_DIM,PHILIP_DIM+2>;


} // Tests namespace
} // PHiLiP namespace

