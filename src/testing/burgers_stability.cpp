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
#include <deal.II/base/convergence_table.h>

#include "burgers_stability.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"
#include <fstream>


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
        
        dealii::LinearAlgebra::distributed::Vector<double> Mu_hat(dg->solution.size());
        dg->global_mass_matrix.vmult( Mu_hat, dg->solution);

	for (unsigned int i = 0; i < dg->solution.size(); ++i)
	{
           // energy += dg->global_mass_matrix(i,i) * dg->solution(i) * dg->solution(i);
            energy +=  dg->solution(i) * Mu_hat(i);
	}
	return energy;
}
template<int dim, int nstate>
void BurgersEnergyStability<dim, nstate>::initialize(PHiLiP::DGBase<dim, double>  &dg) const
{
	pcout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<dim> initial_condition;
	std::string variables;
        if (dim == 3)
	    variables = "x,y,z";
        if (dim == 2)
	    variables = "x,y";
        if (dim == 1)
	    variables = "x";
	//std::string variables = "x";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression;
        if (dim == 3)
	    expression = "sin(pi*(x))*sin(pi*y)*sin(pi*z) + 0.01";
        if (dim == 2)
	    expression = "sin(pi*(x))*sin(pi*y) + 0.01";
        if(dim==1)
	    expression = "sin(pi*(x)) + 0.01";
	initial_condition.initialize(variables,
	                             expression,
	                             constants);
//	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
        dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
        solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
	dealii::VectorTools::interpolate(dg.dof_handler,initial_condition,solution_no_ghost);
        dg.solution = solution_no_ghost;

}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers energy stability. " << std::endl;
//	dealii::Triangulation<dim> grid;

        PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
	double left = 0.0;
	double right = 2.0;
	const unsigned int n_grids = 8;
        std::array<double,n_grids> grid_size;
        std::array<double,n_grids> soln_error;
	unsigned int poly_degree = 5;
        dealii::ConvergenceTable convergence_table;
        const unsigned int igrid_start = 7;

//        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids_input);
        for(unsigned int igrid = igrid_start; igrid<n_grids; igrid++){

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
			dealii::Triangulation<dim> grid(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement |
					dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
			dealii::parallel::distributed::Triangulation<dim> grid(
				this->mpi_communicator);
#endif
//straight
    dealii::GridGenerator::hyper_cube(grid, left, right, true);
   // dealii::GridGenerator::subdivided_hyper_cube(grid,static_cast<int>(pow(2, igrid)),left, right);
      //  dealii::GridGenerator::subdivided_hyper_cube(grid_super_fine, n_1d_cells[n_grids_input-1]);
//curvilinear
#if 0
const dealii::Point<dim> center1(-1,1);
const dealii::SphericalManifold<dim> m0(center1);
dealii::GridGenerator::hyper_cube (grid, left, right, colorize);
grid.set_manifold(0, m0);
const dealii::Point<dim> center2(1,1);
const dealii::SphericalManifold<dim> m02(center2);
grid.set_manifold(1, m02);
for(int idim=0; idim<dim; idim++){
grid.set_all_manifold_ids_on_boundary(2*(idim -1),2*(idim-1));
grid.set_all_manifold_ids_on_boundary(2*(idim -1)+1,2*(idim-1)+1);
}
#endif
#if PHILIP_DIM==1
#else
	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
                if(dim == 2)
		dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
                if(dim==3)
		dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
		grid.add_periodicity(matched_pairs);
#endif
	grid.refine_global(igrid);
	pcout << "Grid generated and refined" << std::endl;
//CFL number
    const unsigned int n_global_active_cells2 = grid.n_global_active_cells();
    double n_dofs_cfl = pow(n_global_active_cells2,dim) * pow(poly_degree+1.0, dim);
    double delta_x = (right-left)/pow(n_dofs_cfl,(1.0/dim)); 
    all_parameters_new.ode_solver_param.initial_time_step =  0.005*delta_x;
  //  all_parameters_new.ode_solver_param.initial_time_step =  0.05*delta_x;
    all_parameters_new.ode_solver_param.initial_time_step =  0.0001;
  //  all_parameters_new.ode_solver_param.initial_time_step =  0.00005;
 //   all_parameters_new.ode_solver_param.initial_time_step =  0.001;
   // all_parameters_new.ode_solver_param.initial_time_step =  0.00001;
  //  if(igrid ==6 )
  //  all_parameters_new.ode_solver_param.initial_time_step =  0.00001;
    
    
         
//allocate dg
//	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, &grid);
	std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, &grid);
	pcout << "dg created" <<std::endl;
	dg->allocate_system ();

        initialize(*(dg));
    #if 0
	pcout << "Implement initial conditions" << std::endl;
	dealii::FunctionParser<dim> initial_condition;
	std::string variables;
        if (dim == 3)
	    variables = "x,y,z";
        if (dim == 2)
	    variables = "x,y";
        if (dim == 1)
	    variables = "x";
	//std::string variables = "x";
	std::map<std::string,double> constants;
	constants["pi"] = dealii::numbers::PI;
	std::string expression;
        if (dim == 3)
	    expression = "sin(pi*(x))*sin(pi*y)*sin(pi*z) + 0.01";
        if (dim == 2)
	    expression = "sin(pi*(x))*sin(pi*y) + 0.01";
        if(dim==1)
	    expression = "sin(pi*(x)) + 0.01";
	initial_condition.initialize(variables,
	                             expression,
	                             constants);
//	dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
	dealii::VectorTools::interpolate(*(dg).dof_handler,initial_condition,dg.solution);
#endif
	// Create ODE solver using the factory and providing the DG object
//	std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
	std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

	double finalTime = 3.0;

        if (all_parameters_new.use_energy == true){//for split form get energy
	    double dt = all_parameters_new.ode_solver_param.initial_time_step;

	//need to call ode_solver before calculating energy because mass matrix isn't allocated yet.

            ode_solver->advance_solution_time(0.000001);
	    double initial_energy = compute_energy(dg);

	//currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
	//this causes some issues with outputs (only one file is output, which is overwritten at each time step)
	//also the ode solver output doesn't make sense (says "iteration 1 out of 1")
	//but it works. I'll keep it for now and need to modify the output functions later to account for this.
	std::ofstream myfile ("energy_plot_cDG_split_p5.gpl" , std::ios::trunc);

	for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	{
		ode_solver->advance_solution_time(dt);
		double current_energy = compute_energy(dg);
                current_energy /=initial_energy;
		pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
		myfile << i * dt << " " << current_energy << std::endl;
		if (current_energy*initial_energy - initial_energy >= 1.0)
		{
			return 1;
			break;
		}
	}
	myfile.close();
        }//end of energy
        else{//do OOA
            finalTime = 1.0;
           // finalTime = 10.0*0.001;
           // finalTime =3.0;

          //  finalTime = 10.0 * all_parameters_new.ode_solver_param.initial_time_step;
	    ode_solver->advance_solution_time(finalTime);
            const unsigned int n_global_active_cells = grid.n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                 << "\t Polynomial degree p: " << poly_degree
                 << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            //dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            //dealii::MappingQ<dim,dim> mappingq(dg->max_degree+1);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;

            // Integrate solution error and output error

            const double pi = atan(1)*4.0;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    for (int istate=0; istate<nstate; ++istate) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact=0.01;
                    for(int idim=0; idim<dim; idim++){
                        uexact += sin(pi*(qpoint[idim]-finalTime));//for grid 1-3
                    }
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                }

            }
#if PHILIP_DIM==1 
            const double l2error_mpi_sum = sqrt(l2error);
#else
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
#endif

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);

            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
              //  const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
     //                                 / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid
                     << "  to grid " << igrid+1
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
            }
        

    
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
      //  convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
     //   convergence_table.set_scientific("output_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());



        }//end of OOA

        //    grid.clear();
        //    grid.~Triangulation();
        // ~Triangulation<dim> grid();
        //    delete grid;
    
        
        }//end of grid loop
        

	return 0; //need to change
}
//#if PHILIP_DIM==1
    template class BurgersEnergyStability<PHILIP_DIM,PHILIP_DIM>;
    //template class BurgersEnergyStability<PHILIP_DIM,1>;
//#endif
} // Tests namespace
} // PHiLiP namespace
