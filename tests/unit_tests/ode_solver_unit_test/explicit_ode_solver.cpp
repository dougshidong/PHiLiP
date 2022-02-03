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
    if (dim == 2){
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
    }
    else if (dim == 1){
        dealii::GridTools::collect_periodic_faces(*grid,
			0, //left
			1, //right
			0, //periodic in x-direction
			matched_pairs);
    }

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
    //all_parameters.ode_solver_param.ode_output = PHiLiP::Parameters::OutputEnum::quiet;    //doesn't seem to do anything
    

    double adv_speed_x = 1.0, adv_speed_y = 0.0;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.advection_vector[0] = adv_speed_x; //x-velocity
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.advection_vector[1] = adv_speed_y; //y-velocity

    int n_time_refinements = 3;
    double dt_init = 0.001;
    double dt = dt_init;
    const double refine_ratio = 0.5;
    double finalTime =2.0;
    
    double L2_error_store[4] {};
    double L2_error; //assigned each loop


    for (int refinement = 0; refinement < n_time_refinements+1; ++refinement){

    //initial_time_step is not modified by explicit ODE solver
    all_parameters.ode_solver_param.initial_time_step = dt;
    std::cout << "Using time step = " << dt << std::endl;
    std::cout << "refinement = " << refinement << std::endl;
    
    all_parameters.ode_solver_param.output_solution_every_x_steps = int(finalTime/dt/10.0); //output 10 vtk files (if dt reaches finalTime exactly)
    
    unsigned int space_poly_degree = 6;
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, space_poly_degree, grid);
    dg->allocate_system ();

    //initial conditions
    std::cout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<dim> initial_condition;
    std::string variables;
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression_initial = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";
    if (dim == 2){
        variables = "x,y";
        expression_initial = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";
    }
    else if (dim == 1){
        variables = "x";
        //expression_initial = "exp(- 20 * (x-1) * (x-1))";	
        expression_initial = "sin(2*pi*x/2.0)";	
    }
    initial_condition.initialize(variables,
		    expression_initial,
	       	    constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);

    // Create ODE solver using the factory and providing the DG object
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

    //Solve
    ode_solver->advance_solution_time(finalTime);
    
    //Defining exact solution
    dealii::FunctionParser<dim> exact_solution;
    constants["a_x"] = adv_speed_x; //CHECK WHERE THIS IS STORED
    constants["a_y"] = adv_speed_y;
    constants["t"] = finalTime;
    std::string expression_exact;
    if (dim == 2){
        expression_exact = "exp( -( 20*(x-1-a_x*t)*(x-1-a_x*t) + 20*(y-1-a_y*t)*(y-1-a_y*t) ) )";
    }
    else if (dim == 1){
        expression_exact = "sin(2*pi*(x-a_x*t)/2.0)";
        //expression_exact = "exp( - 20*(x-1-a_x*t)*(x-1-a_x*t)) ";
    }
    exact_solution.initialize(variables,
    		    expression_exact,
    		    constants);
    
    //Calculating L2 error
    dealii::Vector<double> difference_per_cell(grid->n_active_cells());
    dealii::VectorTools::integrate_difference(dg->dof_handler, 
		    dg->solution, 
		    exact_solution, 
		    difference_per_cell, 
		    dealii::QGauss<dim>(space_poly_degree+1), //check that this is correct polynomial degree
		    dealii::VectorTools::L2_norm);
    L2_error = 
	    dealii::VectorTools::compute_global_error(*grid,
			    difference_per_cell,
			    dealii::VectorTools::L2_norm);
    std::cout << "Computed error is " << L2_error << std::endl;
    std::cout << "Number of cells is "<< grid->n_active_cells()<<std::endl;

    dt *= refine_ratio;
    L2_error_store[refinement] = L2_error;
    }//time refinement loop

    //printing results 
    //should make prettier (use dealii tables)    
    std::cout << "dt  |  L2 norm of error" << std::endl;
    dt = dt_init;   
    for (int refinement = 0; refinement < n_time_refinements+1; ++refinement){
        std::cout << dt << "   |   " << L2_error_store[refinement] << std::endl;
	dt*=refine_ratio;
    }
	    
    return 0; //need to change
}

