// for the grid:
#include "grid_refinement_study.h"

// for the actual test:
#include "flow_solver.h" // includes all required for InitialConditionFunction

#include <deal.II/base/function.h>

#include <stdlib.h>
#include <iostream>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h> // not needed?

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"

#include "mesh/grids/straight_periodic_cube.hpp"

#include <fstream>

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
, number_of_cells_per_direction(all_param.grid_refinement_study_param.grid_size)
, domain_left(all_param.grid_refinement_study_param.grid_left)
, domain_right(all_param.grid_refinement_study_param.grid_right)
, final_time(flow_solver_param.final_time)
, domain_volume(pow(domain_right - domain_left, dim))
, unsteady_data_table_filename("tgv_kinetic_energy_vs_time_table.txt")
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;

    // Flow case identifiers
    is_taylor_green_vortex = ((flow_type == FlowCaseEnum::inviscid_taylor_green_vortex) || (flow_type == FlowCaseEnum::viscous_taylor_green_vortex));

    // Assign the domain boundaries, domain volume, and grid type for each flow case
    // if (is_taylor_green_vortex) {
    //     domain_volume = pow(domain_right - domain_left, dim);
    // }
}

template <int dim, int nstate>
void FlowSolver<dim,nstate>::display_flow_solver_setup(const Parameters::AllParameters *const all_param) const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = all_param->pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    pcout << "- PDE Type: " << pde_string << std::endl;

    std::string flow_type_string;
    if(is_taylor_green_vortex) {
        flow_type_string = "Taylor Green Vortex";
        pcout << "- Flow Case: " << flow_type_string << std::endl;
        pcout << "- - Freestream Reynolds number: " << all_param->navier_stokes_param.reynolds_number_inf << std::endl;
        pcout << "- - Freestream Mach number: " << all_param->euler_param.mach_inf << std::endl;
    }
}

template<int dim, int nstate>
double FlowSolver<dim,nstate>::integrand_kinetic_energy(const std::array<double,nstate> &soln_at_q) const
{
    // Description: Returns nondimensional kinetic energy
    const double nondimensional_density = soln_at_q[0];
    double dot_product_of_nondimensional_momentum = 0.0;
    for (int d=0; d<dim; ++d) {
        dot_product_of_nondimensional_momentum += soln_at_q[d+1]*soln_at_q[d+1];
    }
    const double nondimensional_kinetic_energy = 0.5*(dot_product_of_nondimensional_momentum)/nondimensional_density;
    return nondimensional_kinetic_energy/domain_volume;
}

template<int dim, int nstate>
double FlowSolver<dim,nstate>::integrand_l2_error_initial_condition(const std::array<double,nstate> &soln_at_q, const dealii::Point<dim> qpoint) const
{
    // Description: Returns l2 error with the initial condition function
    // Purpose: For checking the initialization
    double integrand_value = 0.0;
    for (int istate=0; istate<nstate; ++istate) {
        const double exact_soln_at_q = initial_condition_function->value(qpoint, istate);
        integrand_value += pow(soln_at_q[istate] - exact_soln_at_q, 2.0);
    }
    return integrand_value;
}

template<int dim, int nstate>
double FlowSolver<dim, nstate>::integrate_over_domain(DGBase<dim, double> &dg,const std::string integrate_what) const
{
    double integral_value = 0.0; 

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            double integrand_value = 0.0;
            if(integrate_what=="kinetic_energy") {integrand_value = integrand_kinetic_energy(soln_at_q);}
            if(integrate_what=="l2_error_initial_condition") {integrand_value = integrand_l2_error_initial_condition(soln_at_q,qpoint);}


            integral_value += integrand_value * fe_values_extra.JxW(iquad);
        }
    }
    const double integral_value_mpi_sum = dealii::Utilities::MPI::sum(integral_value, mpi_communicator);
    return integral_value_mpi_sum;
}

template <int dim, int nstate>
void FlowSolver<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const unsigned int current_iteration,
    const double current_time, 
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{
    // Add time to table
    std::string time_string = "time";
    unsteady_data_table->add_value(time_string, current_time);
    unsteady_data_table->set_precision(time_string, 16);
    unsteady_data_table->set_scientific(time_string, true);
    // Compute and add kinetic energy to table
    std::string kinetic_energy_string = "kinetic_energy";
    const double kinetic_energy = integrate_over_domain(*dg,"kinetic_energy");
    unsteady_data_table->add_value(kinetic_energy_string, kinetic_energy);
    unsteady_data_table->set_precision(kinetic_energy_string, 16);
    unsteady_data_table->set_scientific(kinetic_energy_string, true);
    // Write to file
    std::ofstream unsteady_data_table_file(unsteady_data_table_filename);
    unsteady_data_table->write_text(unsteady_data_table_file);
    // Print to console
    pcout << "    Iter: " << current_iteration 
          << "    Time: " << current_time 
          << "    Energy: " << kinetic_energy 
          << std::endl;

    // Abort if energy is nan
    if(std::isnan(kinetic_energy)) {
        pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
        pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run_test() const
{
    pcout << "Running Flow Solver... " << std::endl;
    //----------------------------------------------------
    // Display flow solver setup
    //----------------------------------------------------
    pcout << "Flow setup: " << std::endl;    
    display_flow_solver_setup(&all_param);
    pcout << "- Polynomial degree: " << poly_degree << std::endl;
    pcout << "- Courant-Friedrich-Lewy number: " << courant_friedrich_lewy_number << std::endl;
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
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>; // Note: Triangulation == MeshType (true for 2D and 3D)
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator);
    // TO DO: should call a grid factory instead of hard coding for periodic cube
    Grids::straight_periodic_cube<dim,Triangulation>(grid, domain_left, domain_right, number_of_cells_per_direction);
    // Display the information about the grid
    pcout << "\n- GRID INFORMATION:" << std::endl;
    // pcout << "- - Grid type: " << grid_type_string << std::endl;
    pcout << "- - Domain dimensionality: " << dim << std::endl;
    pcout << "- - Domain left: " << domain_left << std::endl;
    pcout << "- - Domain right: " << domain_right << std::endl;
    pcout << "- - Number of cells in each direction: " << number_of_cells_per_direction << std::endl;
    pcout << "- - Domain volume: " << domain_volume << std::endl;
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
    const unsigned int number_of_degrees_of_freedom = dg->dof_handler.n_dofs();
    const double approximate_grid_spacing = (domain_right-domain_left)/pow(number_of_degrees_of_freedom,(1.0/dim));
    const double constant_time_step = courant_friedrich_lewy_number * approximate_grid_spacing;
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
    // - Output initialization to be viewed in Paraview
    dg->output_results_vtk(9999); 
    // - Check L2 error for initialization
    pcout << "Note: L2 error for initialization: " << integrate_over_domain(*dg,"l2_error_initial_condition") << std::endl;
    //----------------------------------------------------
    // ODE Solver
    //----------------------------------------------------
    pcout << "Creating ODE solver... " << std::flush;
    std::shared_ptr<ODE::ODESolverBase<dim, double, Triangulation>> ode_solver = ODE::ODESolverFactory<dim, double, Triangulation>::create_ODESolver(dg);
    ode_solver->allocate_ode_system();
    pcout << "done." << std::endl;
    //----------------------------------------------------
    // Computed quantities at initial time
    //----------------------------------------------------
    std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();//(this->mpi_communicator) ?;
    compute_unsteady_data_and_write_to_table(ode_solver->current_iteration, ode_solver->current_time, dg, unsteady_data_table);
    //----------------------------------------------------
    // Time advancement loop with on-the-fly post-processing
    //----------------------------------------------------
    pcout << "\nPreparing to advance solution in time:" << std::endl;
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
        }
    }
    return 0;
}

#if PHILIP_DIM==3    
    template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

