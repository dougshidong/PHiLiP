// for the grid:
#include "grid_refinement_study.h"

// for the actual test:
#include "flow_solver.h" // includes all required for InitialConditionFunction

#include <deal.II/base/function.h>

#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver_cases/periodic_cube_flow.h"
#include "flow_solver_cases/1D_burgers_rewienski_snapshot.cpp"
#include <deal.II/base/table_handler.h>

namespace PHiLiP {

namespace Tests {
//=========================================================
// FLOW SOLVER TEST CASE -- What runs the test
//=========================================================
template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
, initial_condition_function(InitialConditionFactory<dim,double>::create_InitialConditionFunction(parameters_input, nstate))
, all_param(*(TestsBase::all_parameters))
, flow_solver_param(all_param.flow_solver_param)
, ode_param(all_param.ode_solver_param)
, courant_friedrich_lewy_number(flow_solver_param.courant_friedrich_lewy_number)
, poly_degree(all_param.grid_refinement_study_param.poly_degree)
, final_time(flow_solver_param.final_time)
, unsteady_data_table_filename_with_extension(flow_solver_param.unsteady_data_table_filename+".txt")
{
    // nothing to do here yet
}

template <int dim, int nstate>
void FlowSolver<dim,nstate>::display_flow_solver_setup() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    if (pde_type == PDE_enum::burgers_rewienski)    {pde_string = "burgers_rewienski";}
    pcout << "- PDE Type: " << pde_string << std::endl;
    pcout << "- Polynomial degree: " << poly_degree << std::endl;
    pcout << "- Final time: " << final_time << std::endl;
}

template <int dim, int nstate>
double FlowSolver<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> /*dg*/) const
{
    pcout << "Using initial time step in ODE parameters." <<std::endl;
    return ode_param.initial_time_step;
}

template <int dim, int nstate>
void FlowSolver<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const unsigned int /*current_iteration*/,
    const double /*current_time*/, 
    const std::shared_ptr <DGBase<dim, double>> /*dg*/,
    const std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/) const
{
    // do nothing by default
}

template<int dim, int nstate>
void FlowSolver<dim, nstate>::restart_computation_from_outputted_step(std::shared_ptr <DGBase<dim, double>> /*dg*/) const
{
    // to do
}

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run_test() const
{
    pcout << "Running Flow Solver... " << std::endl;
    //----------------------------------------------------
    // Display flow solver setup
    //----------------------------------------------------
    pcout << "Flow solver setup: " << std::endl;    
    display_flow_solver_setup();
    //----------------------------------------------------
    // Physics
    //----------------------------------------------------
    pcout << "Creating physics object... " << std::flush;
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&all_param);
    pcout << "done." << std::endl;
    //----------------------------------------------------
    // Grid
    //----------------------------------------------------
    pcout << "Generating the grid... " << std::flush;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
        this->mpi_communicator
#endif
        );
    generate_grid(grid);
    pcout << "done." << std::endl;
    //----------------------------------------------------
    // Spatial discretization (Discontinuous Galerkin)
    //----------------------------------------------------
    pcout << "Creating Discontinuous Galerkin object... " << std::flush;
    // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, poly_degree, grid_degree, grid);
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, poly_degree, grid);
    dg->allocate_system();
    pcout << "done." << std::endl;
    //----------------------------------------------------
    // Constant time step based on CFL number
    //----------------------------------------------------
    pcout << "Setting constant time step... " << std::flush;
    const double constant_time_step = get_constant_time_step(dg);
    pcout << "done." << std::endl;
    // ----------------------------------------------------
    // Initialize the solution
    // ----------------------------------------------------
    pcout << "Initializing solution with initial condition function..." << std::flush;
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *initial_condition_function, solution_no_ghost);
    dg->solution = solution_no_ghost;
    pcout << "done." << std::endl;
    //----------------------------------------------------
    // ODE Solver
    //----------------------------------------------------
    pcout << "Creating ODE solver... " << std::flush;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->allocate_ode_system();
    pcout << "done." << std::endl;
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
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
        pcout << "Writing unsteady data computed at initial time... " << std::endl;
        compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);
        pcout << "done." << std::endl;
        //----------------------------------------------------
        // Time advancement loop with on-the-fly post-processing
        //----------------------------------------------------
        pcout << "Advancing solution in time... " << std::endl;
        while(ode_solver->current_time < final_time)
        {
            ode_solver->step_in_time(constant_time_step,false); // pseudotime==false

            // Compute the unsteady quantities, write to the dealii table, and output to file
            compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);

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
    if (flow_type == FlowCaseEnum::taylor_green_vortex) {
        if constexpr (dim==3 && nstate==dim+2) return std::make_unique<PeriodicCubeFlow<dim,nstate>>(parameters_input);
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot) {
        if constexpr (dim==1 && nstate==dim) return std::make_unique<BurgersRewienskiSnapshot<dim,nstate>>(parameters_input);
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

