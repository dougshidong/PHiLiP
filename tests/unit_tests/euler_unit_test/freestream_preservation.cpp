#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/vector_tools.h> // interpolate initial conditions

#include "mesh/grids/curved_periodic_grid.hpp"

#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"

const double FINAL_TIME = 5.0;
const int POLY_DEGREE = 3;
const int GRID_DEGREE = 4;
const int OVERINTEGRATION = 0;
const unsigned int NX_CELL = 2;
const unsigned int NY_CELL = 3;
const unsigned int NZ_CELL = 4;

template<int dim>
int test()
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    int test_error = 0;
    using namespace PHiLiP;
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    parameter_handler.set("pde_type", "euler");
    parameter_handler.set("conv_num_flux", "lax_friedrichs");
    parameter_handler.set("dimension", (long int)dim);
    parameter_handler.set("overintegration", (long int) OVERINTEGRATION);
    parameter_handler.set("use_collocated_nodes", false);
    parameter_handler.enter_subsection("euler");
    parameter_handler.set("mach_infinity", 0.3);
    parameter_handler.set("angle_of_attack", 36.0);
    //parameter_handler.set("side_slip_angle", -13.0);
    parameter_handler.set("side_slip_angle", 0.0);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("ODE solver");
    parameter_handler.set("ode_solver_type", "explicit");
    parameter_handler.set("nonlinear_max_iterations", (long int) 500);
    parameter_handler.set("nonlinear_max_iterations", (long int) 500);
    double U_plus_c = 1.0 + 1/0.3;
    double time_step = (1.0/NX_CELL) / U_plus_c;
    time_step = std::min((1.0/NY_CELL) / U_plus_c, time_step);
    if(dim == 3) time_step = std::min((1.0/NZ_CELL) / U_plus_c, time_step);
    time_step = 0.1 * time_step;
    parameter_handler.set("initial_time_step", time_step);
    parameter_handler.leave_subsection();

    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);

    param.euler_param.parse_parameters (parameter_handler);

    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[0] = NX_CELL;
    n_subdivisions[1] = NY_CELL;
    if (dim == 3) n_subdivisions[2] = NZ_CELL;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    //dealii::Point<dim> p1, p2;
    //for (int d=0; d<dim; ++d) {
    //    p1[d] = 0.0;
    //    p2[d] = 1.0;
    //}
    //const bool colorize = true;
    //dealii::GridGenerator::subdivided_hyper_rectangle (*grid, n_subdivisions, p1, p2, colorize);
    //std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator> > matched_pairs;
    //for (int d=0; d<dim; ++d) {
    //    dealii::GridTools::collect_periodic_faces(*grid,d*2,d*2+1,d,matched_pairs);
    //}
    //grid->add_periodicity(matched_pairs);

    Grids::curved_periodic_sine_grid<dim>(*grid, n_subdivisions);

    pcout << "Number of cells: " << grid->n_active_cells() << std::endl;

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, POLY_DEGREE, GRID_DEGREE, grid);
    dg->allocate_system ();

    // Initialize coarse grid solution with free-stream
    Physics::Euler<dim,dim+2,double> euler_physics_double = Physics::Euler<dim, dim+2, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,dim+2> initial_conditions(euler_physics_double);
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

    const auto initial_constant_solution = dg->solution;

    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->allocate_ode_system();

    //dealii::FullMatrix<double> fullA(dg->global_inverse_mass_matrix.m());
    //fullA.copy_from(dg->global_inverse_mass_matrix);
    //pcout<<"Dense matrix:"<<std::endl;
    //if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);

    int n_steps = FINAL_TIME / time_step;
    pcout << "Time step: " << time_step << std::endl;
    for (int i=0; i < n_steps; ++i) {
        ode_solver->current_iteration = i;
        pcout << " ********************************************************** "
              << std::endl
              << " Iteration: " << ode_solver->current_iteration + 1
              << " out of: " << n_steps
              << std::endl;
        
        dg->assemble_residual();
        pcout << "Residual norm: " << dg->get_residual_l2norm() << std::endl;
        dg->output_results_vtk(i);
        const bool pseudotime = true;
        ode_solver->step_in_time(time_step, pseudotime);
        auto diff = initial_constant_solution;
        diff -= dg->solution;
        double diff_norm = diff.l2_norm();
        pcout << "Solution change norm: " << diff_norm << std::endl;
        if (diff_norm > 1e-3) {
            std::cout << "Freestream flow is not preserved" << std::endl;
            return 1;
        }
    }

    dg->output_results_vtk(9999);

    return test_error;
}


int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test<PHILIP_DIM>();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return test_error;
}


