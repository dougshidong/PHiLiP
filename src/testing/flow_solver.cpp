#include "flow_solver.h"

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
    pcout << "Initializing solution with initial condition function..." << std::flush;
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *initial_condition_function, solution_no_ghost);
    dg->solution = solution_no_ghost;
    pcout << "done." << std::endl;
    ode_solver->allocate_ode_system();
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
        // Constant time step based on CFL number
        //----------------------------------------------------
        pcout << "Setting constant time step... " << std::flush;
        const double constant_time_step = flow_solver_case->get_constant_time_step(dg);
        pcout << "done." << std::endl;
        //----------------------------------------------------
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
        pcout << "Writing unsteady data computed at initial time... " << std::endl;
        flow_solver_case->compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);
        pcout << "done." << std::endl;
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        pcout << "Advancing solution in time... " << std::endl;
        while(ode_solver->current_time < final_time)
        {
            ode_solver->step_in_time(constant_time_step,false); // pseudotime==false

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

