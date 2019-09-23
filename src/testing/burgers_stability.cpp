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

#include "burgers_stability.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersEnergyStability<dim, nstate>::BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
{
	double energy = 0.0;
	for (unsigned int i = 0; i < dg->solution.size(); ++i)
	{
		energy += 1./(dg->global_inverse_mass_matrix(i,i)) * dg->solution(i) * dg->solution(i);
	}
	return energy;
}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers energy stability. " << std::endl;
//	dealii::Triangulation<dim> grid(
//	                typename dealii::Triangulation<dim>::MeshSmoothing(
//	                    dealii::Triangulation<dim>::smoothing_on_refinement |
//	                    dealii::Triangulation<dim>::smoothing_on_coarsening));
	dealii::Triangulation<dim> grid;

	double left = 0.0;
	double right = 2.0;
	const bool colorize = true;
	int n_refinements = 5;
	unsigned int poly_degree = 7;
	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);
	grid.refine_global(n_refinements);
	pcout << "Grid generated and refined" << std::endl;
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
	pcout << "dg created" <<std::endl;
	dg->allocate_system ();

	pcout << "Implement initial conditions" << std::endl;
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

	double finalTime = 3.;

	double dt = all_parameters->ode_solver_param.initial_time_step;
	//(void) dt;

	//need to call ode_solver before calculating energy because mass matrix isn't allocated yet.

	ode_solver->advance_solution_time(0.000001);
	double initial_energy = compute_energy(dg);

	//currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
	//this causes some issues with outputs (only one file is output, which is overwritten at each time step)
	//also the ode solver output doesn't make sense (says "iteration 1 out of 1")
	//but it works. I'll keep it for now and need to modify the output functions later to account for this.
	std::ofstream myfile ("energy_plot.gpl" , std::ios::trunc);

	for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	{
		ode_solver->advance_solution_time(dt);
		double current_energy = compute_energy(dg);
		pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
		myfile << i * dt << " " << current_energy << std::endl;
		if (current_energy - initial_energy >= 0.001)
		{
			return 1;
			break;
		}
	}
	myfile.close();


	//ode_solver->advance_solution_time(finalTime);

	return 0; //need to change
}
#if PHILIP_DIM==1
    template class BurgersEnergyStability<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
