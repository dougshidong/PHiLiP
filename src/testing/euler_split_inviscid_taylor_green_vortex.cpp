#include "euler_split_inviscid_taylor_green_vortex.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreen<dim, nstate>::EulerTaylorGreen(const Parameters::AllParameters *const parameters_input)
:
TestsBase::TestsBase(parameters_input)
{}

//template <int dim, int nstate>
//double EulerTaylorGreen<dim,nstate>::compute_quadrature_kinetic_energy(std::array<double,nstate> soln_at_q) const
//{
//	const double density = soln_at_q[0];
//
//	const double quad_kin_energ =  0.5*(soln_at_q[1]*soln_at_q[1] +
//										soln_at_q[2]*soln_at_q[2] +
//										soln_at_q[3]*soln_at_q[3] )/density;
//	return quad_kin_energ;
//}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
	// Overintegrate the error to make sure there is not integration error in the error estimate
	int overintegrate = 10 ;//10;
	dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
	            dealii::FEValues<dim,dim> fe_values_extra(dealii::MappingQ<dim>(dg->max_degree+overintegrate), dg->fe_collection[poly_degree], quad_extra,
	            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
//	dealii::QGauss<dim> quad_extra(dg->fe_system.tensor_degree()+overintegrate);
//	dealii::FEValues<dim,dim> fe_values_extra(dg->mapping, dg->fe_system, quad_extra,
//							dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
	const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
	std::array<double,nstate> soln_at_q;

	double total_kinetic_energy = 0;

	// Integrate solution error and output error
	typename dealii::DoFHandler<dim>::active_cell_iterator
	cell = dg->dof_handler.begin_active(),
	endc = dg->dof_handler.end();

	std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

	            //const double gam = euler_physics_double.gam;
	            //const double mach_inf = euler_physics_double.mach_inf;
	            //const double tot_temperature_inf = 1.0;
	            //const double tot_pressure_inf = 1.0;
	            //// Assuming a tank at rest, velocity = 0, therefore, static pressure and temperature are same as total
	            //const double density_inf = gam*tot_pressure_inf/tot_temperature_inf * mach_inf * mach_inf;
	            //const double entropy_inf = tot_pressure_inf*pow(density_inf,-gam);
	//const double entropy_inf = euler_physics_double.entropy_inf;

	for (; cell!=endc; ++cell) {

		fe_values_extra.reinit (cell);
		//std::cout << "sitting on cell " << cell->index() << std::endl;
	    cell->get_dof_indices (dofs_indices);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        	std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
	        for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
	        	const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
	            soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
	        }

	        const double density = soln_at_q[0];

	        const double quadrature_kinetic_energy =  0.5*(soln_at_q[1]*soln_at_q[1] +
	        										soln_at_q[2]*soln_at_q[2] +
	        										soln_at_q[3]*soln_at_q[3] )/density;

	        //const double quadrature_kinetic_energy = compute_quadrature_kinetic_energy(soln_at_q);

	        total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
	}
	return total_kinetic_energy;
}

template <int dim, int nstate>
int EulerTaylorGreen<dim, nstate>::run_test() const
{
	//dealii::Triangulation<dim> grid;
	dealii::parallel::distributed::Triangulation<dim> grid(mpi_communicator);

	double left = 0.0;
	double right = 2 * dealii::numbers::PI;
	const bool colorize = true;
	int n_refinements = 2;
	unsigned int poly_degree = 1;
	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);

	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
	dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
	dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
	dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
	grid.add_periodicity(matched_pairs);

	grid.refine_global(n_refinements);

	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
	dg->allocate_system ();

	std::cout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<PHILIP_DIM> initial_condition(5);
	std::string variables = "x,y,z";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::vector<std::string> expressions(5);
	expressions[0] = "1";
	expressions[1] = "sin(x)*cos(y)*cos(z)";
	expressions[2] = "-cos(x)*sin(y)*cos(z)";
	expressions[3] = "0";
	expressions[4] = "250.0/1.4 + 2.5/16.0 * (cos(2.0*x)*cos(2.0*z) + 2.0*cos(2.0*y) + 2.0*cos(2.0*x) + cos(2.0*y)*cos(2.0*z)) + 0.5 * pow(cos(z),2.0) * (pow(cos(x),2.0) * pow(sin(y),2.0) +pow(sin(x),2.0) * pow(cos(y),2.0))";
	initial_condition.initialize(variables,
	                             expressions,
	                             constants);
//	dealii::Point<3> point (0.0,1.0,1.0);
//	dealii::Vector<double> result(5);
//	initial_condition.vector_value(point,result);
//	dealii::deallog << result;

	std::cout << "initial condition successfully implemented" << std::endl;
	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	std::cout << "initial condition interpolated to DG solution" << std::endl;
	// Create ODE solver using the factory and providing the DG object

	std::cout << "creating ODE solver" << std::endl;
	std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
	std::cout << "ODE solver successfully created" << std::endl;
	double finalTime = 14.;
	double dt = all_parameters->ode_solver_param.initial_time_step;

	//double dt = all_parameters->ode_solver_param.initial_time_step;
	//need to call ode_solver before calculating energy because mass matrix isn't allocated yet.
	//ode_solver->advance_solution_time(0.000001);
	//double initial_energy = compute_energy(dg);

	std::cout << "preparing to advance solution in time" << std::endl;
//	//(void) finalTime;
//	std::cout << "kinetic energy is" << compute_kinetic_energy(dg) <<std::endl;
//	ode_solver->advance_solution_time(finalTime);
//	//std::cout << "kinetic energy is" << compute_kinetic_energy(dg) <<std::endl;
//	std::cout << "kinetic energy is" << compute_kinetic_energy(dg) <<std::endl;


	//currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
	//this causes some issues with outputs (only one file is output, which is overwritten at each time step)
	//also the ode solver output doesn't make sense (says "iteration 1 out of 1")
	//but it works. I'll keep it for now and need to modify the output functions later to account for this.

	std::ofstream myfile ("kinetic_energy_plot.gpl" , std::ios::trunc);

	for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	{
		ode_solver->advance_solution_time(dt);
		double current_energy = compute_kinetic_energy(dg,poly_degree);
		std::cout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
		myfile << i * dt << " " << current_energy << std::endl;

	}

	myfile.close();

//	ode_solver->advance_solution_time(finalTime);
//	(void) dt;
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
//        // Declare possible inputs
//        dealii::ParameterHandler parameter_handler;
//        Parameters::AllParameters::declare_parameters (parameter_handler);
//        Parameters::parse_command_line (argc, argv, parameter_handler);
//
//        // Read inputs from parameter file and set those values in AllParameters object
//        Parameters::AllParameters all_parameters;
//        std::cout << "Reading input..." << std::endl;
//        all_parameters.parse_parameters (parameter_handler);
//
//        AssertDimension(all_parameters.dimension, PHILIP_DIM);
//
//        std::cout << "Starting program..." << std::endl;
//
//		using namespace PHiLiP;
//		//const Parameters::AllParameters parameters_input;
//		EulerTaylorGreen<PHILIP_DIM, PHILIP_DIM+2> euler_test(&all_parameters);
//		int i = euler_test.run_test();
//		return i;
//	}
//	catch (std::exception &exc)
//	{
//		std::cerr << std::endl << std::endl
//				  << "----------------------------------------------------"
//				  << std::endl
//				  << "Exception on processing: " << std::endl
//	          	  << exc.what() << std::endl
//	          	  << "Aborting!" << std::endl
//	          	  << "----------------------------------------------------"
//	          	  << std::endl;
//		return 1;
//	}
//
//	catch (...)
//	{
//	    std::cerr << std::endl
//	              << std::endl
//	              << "----------------------------------------------------"
//	              << std::endl
//	              << "Unknown exception!" << std::endl
//	              << "Aborting!" << std::endl
//	              << "----------------------------------------------------"
//	              << std::endl;
//	    return 1;
//	}
//	std::cout << "End of program." << std::endl;
//	return test_error;
//}

#if PHILIP_DIM==3
    template class EulerTaylorGreen <PHILIP_DIM,PHILIP_DIM+2>;
#endif

}
}



