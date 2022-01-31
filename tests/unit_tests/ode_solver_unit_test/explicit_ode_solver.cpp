#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "numerical_flux/numerical_flux_factory.hpp"
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"


int main (int argc, char * argv[])
{
dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
const int dim = PHILIP_DIM;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
    MPI_COMM_WORLD,
#endif
    typename dealii::Triangulation<dim>::MeshSmoothing(
    dealii::Triangulation<dim>::smoothing_on_refinement |
    dealii::Triangulation<dim>::smoothing_on_coarsening));

    double left = 0.0;
    double right = 2.0;
    const bool colorize = true;
    int n_refinements = 5;

    //generating grid
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
    //setting periodic BC 
    //to do: change this to work with 1D
    std::vector<dealii::GridTools::PeriodicFacePair<typename Triangulation::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
    dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
    grid->add_periodicity(matched_pairs);

    grid->refine_global(n_refinements);
    
    //default parameters
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler); // default fills options
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler); // copies stuff from parameter_handler into all_parameters

    //parameters consistent with MPI_2D_ADVECTION_EXPLICIT_PERIODIC_LONG test
    all_parameters.ode_solver_param.ode_solver_type = PHiLiP::Parameters::ODESolverParam::ODESolverEnum::explicit_solver;
    all_parameters.ode_solver_param.nonlinear_max_iterations = 500;
    all_parameters.ode_solver_param.print_iteration_modulo = 100;
    
    const double dt = 0.01; // later refine this 

    //initial_time_step is not modified by explicit ODE solver
    all_parameters.ode_solver_param.initial_time_step = dt;

    unsigned int space_poly_degree = 5;
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, space_poly_degree, grid);
    dg->allocate_system ();

    //initial conditions
    //to do: make 1D ICs
#if PHILIP_DIM == 2	
    std::cout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<2> initial_condition;
    std::string variables = "x,y";
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression_initial = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";
    initial_condition.initialize(variables,
    expression_initial,
    constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
#endif
    
    // Create ODE solver using the factory and providing the DG object
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

    double finalTime = 0.1;

    //double dt = all_parameters->ode_solver_param.initial_time_step;
    ode_solver->advance_solution_time(finalTime);
    
    //to do:
    //add error handling
    //	define exact solution
    //	https://www.dealii.org/current/doxygen/deal.II/step_7.html is example for convergence error
    //		VectorTools::integrate_difference()
    //		VectorTools::compute_global_error()
    //	eventually also format into table
    //
    //
   const double advection_speed = 1.0;

#if PHILIP_DIM == 2
    dealii::FunctionParser<2> exact_solution;
    constants["a"] = advection_speed;
    constants["t"] = finalTime;
    std::string expression_exact = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";
    exact_solution.initialize(variables,
		    expression_exact,
		    constants);

#endif


    return 0; //need to change
}

