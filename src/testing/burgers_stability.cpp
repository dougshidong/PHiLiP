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
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/initial_condition.h"


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
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
        if(dg->all_parameters->use_inverse_mass_on_the_fly)
            dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
        else
            dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);
        //Since we normalize the energy later, don't bother scaling by 0.5
        //Energy \f$ = 0.5 * \int u^2 d\Omega_m \f$
        energy = dg->solution * mass_matrix_times_solution;
        
	return energy;
}

template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const
{
        //Conservation \f$ =  \int 1 * u d\Omega_m \f$
	double conservation = 0.0;
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
        if(dg->all_parameters->use_inverse_mass_on_the_fly)
            dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
        else
            dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        std::vector<double> ones(n_quad_pts, 1.0);
        //Projected vector of ones. That is, the interpolation of ones_hat to the volume nodes is 1.
        std::vector<double> ones_hat(n_dofs_cell);
        //We have to project the vector of ones because the mass matrix has an interpolation from solution nodes built into it.
        OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(dg->nstate, poly_degree, dg->max_grid_degree);
        vol_projection.build_1D_volume_operator(dg->oneD_fe_collection[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
        vol_projection.matrix_vector_mult_1D(ones, ones_hat,
                                                   vol_projection.oneD_vol_operator);

        dealii::LinearAlgebra::distributed::Vector<double> ones_hat_global(dg->right_hand_side);
        std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices (dofs_indices);
            for(unsigned int idof=0;idof<n_dofs_cell; idof++){
                ones_hat_global[dofs_indices[idof]] = ones_hat[idof];
            }
        }

        conservation = ones_hat_global * mass_matrix_times_solution;

	return conservation;
}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers energy stability. " << std::endl;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2.0;
    const unsigned int n_grids = (all_parameters_new.use_energy) ? 4 : 5;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    unsigned int poly_degree = 4;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 3;
    const unsigned int grid_degree = 1;

    for(unsigned int igrid = igrid_start; igrid<n_grids; igrid++){

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        using Triangulation = dealii::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
        //straight grid setup
        dealii::GridGenerator::hyper_cube(*grid, left, right, true);
        //found the periodicity in dealii doesn't work as expected in 1D so I hard coded the 1D periodic condition in DG
#if PHILIP_DIM==1
#else
	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
                if(dim >= 2)
		dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
                if(dim>=3)
		dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
		grid->add_periodicity(matched_pairs);
#endif
	grid->refine_global(igrid);
	pcout << "Grid generated and refined" << std::endl;
        //CFL number
        const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
        double n_dofs_cfl = pow(n_global_active_cells2,dim) * pow(poly_degree+1.0, dim);
        double delta_x = (right-left)/pow(n_dofs_cfl,(1.0/dim)); 
        all_parameters_new.ode_solver_param.initial_time_step =  0.5*delta_x;
        //use 0.0001 to be consisitent with Ranocha and Gassner papers
        all_parameters_new.ode_solver_param.initial_time_step =  0.0001;
        
        
             
        //allocate dg
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        pcout << "dg created" <<std::endl;
        dg->allocate_system ();
         
        //initialize IC
        InitialCondition<dim,nstate,double> initial_condition(dg, &all_parameters_new);
	// Create ODE solver using the factory and providing the DG object
	std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

	double finalTime = 3.0;

        if (all_parameters_new.use_energy == true){//for split form get energy

	    double dt = all_parameters_new.ode_solver_param.initial_time_step;

	    //need to call ode_solver before calculating energy because mass matrix isn't allocated yet.
            ode_solver->current_iteration = 0;
            ode_solver->advance_solution_time(0.000001);
	    double initial_energy = compute_energy(dg);
	    double initial_conservation = compute_conservation(dg, poly_degree);

	    //currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
	    //this causes some issues with outputs (only one file is output, which is overwritten at each time step)
	    //also the ode solver output doesn't make sense (says "iteration 1 out of 1")
	    //but it works. I'll keep it for now and need to modify the output functions later to account for this.
	    std::ofstream myfile ("energy_plot_burgers.gpl" , std::ios::trunc);
             
            ode_solver->current_iteration = 0;
	    for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
	    {
	            ode_solver->advance_solution_time(dt);
                    //Energy
	            double current_energy = compute_energy(dg);
                    current_energy /=initial_energy;
                    std::cout << std::setprecision(16) << std::fixed;
	            pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
	            myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_energy << std::endl;
	            if (current_energy*initial_energy - initial_energy >= 1.0)
	            {
                        pcout<<"Energy not monotonicaly decreasing"<<std::endl;
	            	return 1;
	            	break;
	            }
//	            if ( (current_energy*initial_energy - initial_energy >= 1.0e-11)&&(all_parameters_new.conv_num_flux_type == Parameters::AllParameters::ConvectiveNumericalFlux::entropy_cons_flux) )
//	            {
//                        pcout<<"Energy not conserved"<<std::endl;
//	            	return 1;
//	            	break;
//	            }
                    //Conservation
	            double current_conservation = compute_conservation(dg, poly_degree);
                    current_conservation /=initial_conservation;
                    std::cout << std::setprecision(16) << std::fixed;
	            pcout << "Normalized Conservation at time " << i * dt << " is " << current_conservation<< std::endl;
	            myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_conservation << std::endl;
	            if (current_conservation*initial_conservation - initial_conservation >= 10.00)
	            {
                        pcout << "Not conserved" << std::endl;
	            	return 1;
	            	break;
	            }
	    }
	    myfile.close();
             
            //Print to a file the final solution vs x to plot
	    std::ofstream myfile2 ("solution_burgers.gpl" , std::ios::trunc);
             
            dealii::QGaussLobatto<dim> quad_extra(dg->max_degree+1);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;
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
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                std::cout << std::setprecision(16) << std::fixed;
	        myfile2<< std::fixed << std::setprecision(16) << qpoint[0] << std::fixed << std::setprecision(16) <<" " << soln_at_q[0]<< std::endl;
                }
            
            }
             
	    myfile2.close();
        }//end of energy
        else{//do OOA
            finalTime = 0.001;//This is sufficient for verification

            ode_solver->current_iteration = 0;

	    ode_solver->advance_solution_time(finalTime);
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
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
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
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
                    double uexact=0.0;
                    for(int idim=0; idim<dim; idim++){
                        uexact += cos(pi*(qpoint[idim]-finalTime));//for grid 1-3
                    }
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                }

            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

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
                pcout << "From grid " << igrid
                     << "  to grid " << igrid+1
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
                if(igrid == n_grids-1){
                    if(std::abs(slope_soln_err-(poly_degree+1))>0.05){
                        return 1;
                    }
                }
            }
        

    
            pcout << " ********************************************"
                << std::endl
                << " Convergence rates for p = " << poly_degree
                << std::endl
                << " ********************************************"
                << std::endl;
            convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("dx", true);
            convergence_table.set_scientific("soln_L2_error", true);
            if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        }//end of OOA
    
    }//end of grid loop
        

    return 0; //if got to here means passed the test, otherwise would've failed earlier
}
template class BurgersEnergyStability<PHILIP_DIM,PHILIP_DIM>;
} // Tests namespace
} // PHiLiP namespace

