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
#include "advection_explicit_periodic.h"

#include<fenv.h>

namespace PHiLiP {
namespace Tests {
template <int dim, int nstate>
AdvectionPeriodic<dim, nstate>::AdvectionPeriodic(const PHiLiP::Parameters::AllParameters *const parameters_input)
:
TestsBase::TestsBase(parameters_input)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}


template <int dim, int nstate>
int AdvectionPeriodic<dim, nstate>::run_test() const
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
			dealii::Triangulation<dim> grid(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement |
					dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
			dealii::parallel::distributed::Triangulation<dim> grid(
				this->mpi_communicator);
#endif

	double left = 0.0;
	double right = 2.0;
	const bool colorize = true;
	int n_refinements = 5;
	unsigned int poly_degree = 2;
	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);

	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
		dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
		//dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
		grid.add_periodicity(matched_pairs);

	grid.refine_global(n_refinements);

	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
	dg->allocate_system ();

//	for (auto current_cell = dg->dof_handler.begin_active(); current_cell != dg->dof_handler.end(); ++current_cell) {
//		 if (!current_cell->is_locally_owned()) continue;
//
//		 dg->fe_values_volume.reinit(current_cell);
//		 int cell_index = current_cell->index();
//		 std::cout << "cell number " << cell_index << std::endl;
//		 for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<PHILIP_DIM>::faces_per_cell; ++face_no)
//		 {
//			 if (current_cell->face(face_no)->at_boundary())
//		     {
//				 std::cout << "face " << face_no << " is at boundary" << std::endl;
//		         typename dealii::DoFHandler<PHILIP_DIM>::active_cell_iterator neighbor = current_cell->neighbor_or_periodic_neighbor(face_no);
//		         std::cout << "the neighbor is " << neighbor->index() << std::endl;
//		     }
//		 }
//
//	}

	std::cout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<2> initial_condition;
	std::string variables = "x,y";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression = "exp( -( 20*(x-1)*(x-1) + 20*(y-1)*(y-1) ) )";//"sin(pi*x)*sin(pi*y)";
	initial_condition.initialize(variables,
								 expression,
								 constants);
	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	// Create ODE solver using the factory and providing the DG object
	std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

	double finalTime = 1.5;

	//double dt = all_parameters->ode_solver_param.initial_time_step;
	ode_solver->advance_solution_time(finalTime);

	return 0; //need to change
}

//int main (int argc, char * argv[])
//{
//	//parse parameters first
//	feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
//	dealii::deallog.depth_console(99);
//		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
//		const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
//		const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
//		dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
//		pcout << "Starting program with " << n_mpi << " processors..." << std::endl;
//		if ((PHILIP_DIM==1) && !(n_mpi==1)) {
//			std::cout << "********************************************************" << std::endl;
//			std::cout << "Can't use mpirun -np X, where X>1, for 1D." << std::endl
//					  << "Currently using " << n_mpi << " processors." << std::endl
//					  << "Aborting..." << std::endl;
//			std::cout << "********************************************************" << std::endl;
//			std::abort();
//		}
//	int test_error = 1;
//	try
//	{
//		// Declare possible inputs
//		dealii::ParameterHandler parameter_handler;
//		PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
//		PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);
//
//		// Read inputs from parameter file and set those values in AllParameters object
//		PHiLiP::Parameters::AllParameters all_parameters;
//		std::cout << "Reading input..." << std::endl;
//		all_parameters.parse_parameters (parameter_handler);
//
//		AssertDimension(all_parameters.dimension, PHILIP_DIM);
//
//		std::cout << "Starting program..." << std::endl;
//
//		using namespace PHiLiP;
//		//const Parameters::AllParameters parameters_input;
//		AdvectionPeriodic<PHILIP_DIM, 1> advection_test(&all_parameters);
//		int i = advection_test.run_test();
//		return i;
//	}
//	catch (std::exception &exc)
//	{
//		std::cerr << std::endl << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Exception on processing: " << std::endl
//				  << exc.what() << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//
//	catch (...)
//	{
//		std::cerr << std::endl
//				  << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Unknown exception!" << std::endl
//				  << "Aborting!" << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl;
//		return 1;
//	}
//	std::cout << "End of program." << std::endl;
//	return test_error;
//}

#if PHILIP_DIM==2
    template class AdvectionPeriodic <PHILIP_DIM,1>;
#endif

} //Tests namespace
} //PHiLiP namespace







