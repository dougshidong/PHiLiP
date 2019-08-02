#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "numerical_flux/numerical_flux.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include<fenv.h>

//using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
//using ConvType = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
//using DissType = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;
//
//
//const double TOLERANCE = 1E-12;


template <int dim, int nstate>
class BurgersEnergyStability
{
public:
	BurgersEnergyStability() = delete;
	BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input);
	int run_test();




private:
	double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg);
    const PHiLiP::Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

template <int dim, int nstate>
BurgersEnergyStability<dim, nstate>::BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input)
:
all_parameters(parameters_input)
{}

template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg)
{
	double energy = 0.0;
	for (unsigned int i = 0; i < dg->solution.size(); ++i)
	{
		energy += 1./(dg->global_inverse_mass_matrix(i,i)) * dg->solution(i) * dg->solution(i);
	}
	return energy;
}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test()
{
	dealii::Triangulation<dim> grid;

	double left = 0.0;
	double right = 2.0;
	const bool colorize = true;
	int n_refinements = 5;
	unsigned int poly_degree = 7;
	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);
	grid.refine_global(n_refinements);

	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree);
	dg->set_triangulation(&grid);
	dg->allocate_system ();

	std::cout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<1> initial_condition;
	std::string variables = "x";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression = "sin(pi*(x)) + 0.01";
	initial_condition.initialize(variables,
	                             expression,
	                             constants);
	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	// Create ODE solver using the factory and providing the DG object
	std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

	double finalTime = 1.;

	double dt = all_parameters->ode_solver_param.initial_time_step;
	//need to call ode_solver before calculating energy because mass matrix isn't allocated yet.
	ode_solver->advance_solution_time(0.000001);
	double initial_energy = compute_energy(dg);

	//currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
	//this causes some issues with outputs (only one file is output, which is overwritten at each time step)
	//also the ode solver output doesn't make sense (says "iteration 1 out of 1")
	//but it works. I'll keep it for now and need to modify the output functions later to account for this.
	for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	{
		ode_solver->advance_solution_time(dt);
		double current_energy = compute_energy(dg);
		std::cout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
		if (current_energy - initial_energy >= 0.001)
		{
			return 1;
		}
	}
	return 0; //need to change
}

int main (int argc, char * argv[])
{
	//parse parameters first
	feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
	dealii::deallog.depth_console(99);
	int test_error = 1;
	try
	{
        // Declare possible inputs
        dealii::ParameterHandler parameter_handler;
        PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
        PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);

        // Read inputs from parameter file and set those values in AllParameters object
        PHiLiP::Parameters::AllParameters all_parameters;
        std::cout << "Reading input..." << std::endl;
        all_parameters.parse_parameters (parameter_handler);

        AssertDimension(all_parameters.dimension, PHILIP_DIM);

        std::cout << "Starting program..." << std::endl;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		using namespace PHiLiP;
		//const Parameters::AllParameters parameters_input;
		BurgersEnergyStability<PHILIP_DIM, 1> burgers_test(&all_parameters);
		int i = burgers_test.run_test();
		return i;
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				  << "----------------------------------------------------"
				  << std::endl
				  << "Exception on processing: " << std::endl
	          	  << exc.what() << std::endl
	          	  << "Aborting!" << std::endl
	          	  << "----------------------------------------------------"
	          	  << std::endl;
		return 1;
	}

	catch (...)
	{
	    std::cerr << std::endl
	              << std::endl
	              << "----------------------------------------------------"
	              << std::endl
	              << "Unknown exception!" << std::endl
	              << "Aborting!" << std::endl
	              << "----------------------------------------------------"
	              << std::endl;
	    return 1;
	}
	std::cout << "End of program." << std::endl;
	return test_error;
}





