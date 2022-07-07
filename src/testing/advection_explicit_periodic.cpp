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

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "advection_explicit_periodic.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"

#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"

#include<fenv.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <fstream>

namespace PHiLiP {
namespace Tests {
template <int dim, int nstate>
AdvectionPeriodic<dim, nstate>::AdvectionPeriodic(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
double AdvectionPeriodic<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
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
double AdvectionPeriodic<dim, nstate>::compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const
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
        // std::vector<double> ones(n_quad_pts);
        // for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        //     ones[iquad] = 1.0;
        // }
        // Projected vector of ones. That is, the interpolation of ones_hat to the volume nodes is 1.
        std::vector<double> ones_hat(n_dofs_cell);
        // We have to project the vector of ones because the mass matrix has an interpolation from solution nodes built into it.
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
int AdvectionPeriodic<dim, nstate>::run_test() const
{

    printf("starting test\n");
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  

    const unsigned int n_grids = (all_parameters_new.use_energy) ? 4 : 5;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    std::vector<double> soln_error_inf(n_grids);
    using ADtype = double;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const double left = -1.0;
    const double right = 1.0;
    unsigned int n_refinements = n_grids;
    unsigned int poly_degree = 3;
    unsigned int grid_degree = poly_degree;
    
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 3;

    for(unsigned int igrid=igrid_start; igrid<n_refinements; ++igrid){
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

        //set the warped grid
        PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, igrid);

        //CFL number
        const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
        double n_dofs_cfl = pow(n_global_active_cells2,dim) * pow(poly_degree+1.0, dim);
        double delta_x = (right-left)/pow(n_dofs_cfl,(1.0/dim)); 
        all_parameters_new.ode_solver_param.initial_time_step =  delta_x /(1.0*(2.0*poly_degree+1)) ;
        all_parameters_new.ode_solver_param.initial_time_step =  (all_parameters_new.use_energy) ? 0.05*delta_x : 0.5*delta_x;
        std::cout << "dt " <<all_parameters_new.ode_solver_param.initial_time_step <<  std::endl;
        std::cout << "cells " <<n_global_active_cells2 <<  std::endl;

        //Set the DG spatial sys
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system ();

        std::cout << "Implement initial conditions" << std::endl;
        // Create initial condition function
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new); 
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        // Create ODE solver using the factory and providing the DG object
        std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        double finalTime = 2.0;
    	
        // need to call ode_solver before calculating energy because mass matrix isn't allocated yet.

        if (all_parameters_new.use_energy == true){//for split form get energy
            double dt = all_parameters_new.ode_solver_param.initial_time_step;
             
            ode_solver->current_iteration = 0;
             
            // advance by small amount, basically 0
            ode_solver->advance_solution_time(dt/10.0);
            // compute the initial energy and conservation
            double initial_energy = compute_energy(dg);
            double initial_conservation = compute_conservation(dg, poly_degree);
             
            // create file to write energy and conservation results
            // outputs results as Time, Energy Newline Time, Conservation
            // And energy and conservation values are normalized by the initial value.
            std::ofstream myfile ("energy_plot.gpl" , std::ios::trunc);
             
            ode_solver->current_iteration = 0;
             
            // loop over time steps because needs to evaluate energy and conservation at each time step.
            for (int i = 0; i < std::ceil(finalTime/dt); ++ i){
                
                ode_solver->advance_solution_time(dt);
                //energy
                double current_energy = compute_energy(dg);
                current_energy /=initial_energy;
                std::cout << std::setprecision(16) << std::fixed;
                this->pcout << "Energy at time " << i * dt << " is " << current_energy<< std::endl;
                myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_energy << std::endl;
                if (current_energy*initial_energy - initial_energy >= 1.00)//since normalized by initial
                {
                    this->pcout << " Energy was not monotonically decreasing" << std::endl;
                	return 1;
                }
                if ( (current_energy*initial_energy - initial_energy >= 1.0e-12) && (all_parameters_new.conv_num_flux_type == Parameters::AllParameters::ConvectiveNumericalFlux::central_flux))
                {
                    this->pcout << " Energy was not conserved" << std::endl;
                	return 1;
                }

                // Conservation
                double current_conservation = compute_conservation(dg, poly_degree);
                current_conservation /=initial_conservation;
                std::cout << std::setprecision(16) << std::fixed;
                this->pcout << "Normalized Conservation at time " << i * dt << " is " << current_conservation<< std::endl;
                myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_conservation << std::endl;
                if (current_conservation*initial_conservation - initial_conservation >= 10.00)
                // if (current_energy - initial_energy >= 10.00)
                {
                    this->pcout << "Not conserved" << std::endl;
                	return 1;
                }
             
            }
             
            // Close the file
            myfile.close();
        }
        else{// do OOA
            if(left==-1){
                finalTime = 0.5;
            }
            if(left==0){
                finalTime=1.0;
            }

            ode_solver->current_iteration = 0;

            // advance solution until the final time
            ode_solver->advance_solution_time(finalTime);

            // output results
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            this->pcout << "Dimension: " << dim
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

            // Integrate every cell and compute L2 and Linf errors.
            const double pi = atan(1)*4.0;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            double linf_error = 0.0;
            const dealii::Tensor<1,3,double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
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
                        double uexact=1.0;
                        for(int idim=0; idim<dim; idim++){
                            if(left==0){
                                uexact *= sin(2.0*pi*(qpoint[idim]-adv_speeds[idim]*finalTime));//for grid 1-3
                            }
                            if(left==-1){
                                uexact *= sin(pi*(qpoint[idim]- adv_speeds[idim]*finalTime));//for grid 1-3
                            }
                        }
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                         
                        double inf_temp = std::abs(soln_at_q[istate]-uexact);
                        //store pointwise inf error
                        if(inf_temp > linf_error){
                            linf_error = inf_temp;
                        }
                    }
                }

            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, this->mpi_communicator));
            const double linferror_mpi= (dealii::Utilities::MPI::max(linf_error, this->mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            soln_error_inf[igrid] = linferror_mpi;
            // output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("soln_Linf_error", linferror_mpi);
            // convergence_table.add_value("output_error", output_error[igrid]);


            this->pcout << " Grid size h: " << dx 
                        << " L2-soln_error: " << l2error_mpi_sum
                        << " Linf-soln_error: " << linferror_mpi
                        << " Residual: " << ode_solver->residual_norm
                        << std::endl;

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                const double slope_soln_err_inf = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                // const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
                //                                 / log(grid_size[igrid]/grid_size[igrid-1]);
                this->pcout << "From grid " << igrid-1
                            << "  to grid " << igrid
                            << "  dimension: " << dim
                            << "  polynomial degree p: " << poly_degree
                            << std::endl
                            << "  solution_error1 " << soln_error[igrid-1]
                            << "  solution_error2 " << soln_error[igrid]
                            << "  slope " << slope_soln_err
                            << "  solution_error1_inf " << soln_error_inf[igrid-1]
                            << "  solution_error2_inf " << soln_error_inf[igrid]
                            << "  slope " << slope_soln_err_inf
                            << std::endl;
            
                if(igrid == n_grids-1){
                    if(std::abs(slope_soln_err_inf-(poly_degree+1))>0.1)
                        return 1;
                }            
            }
        }//end of OOA else statement
        this->pcout << " ********************************************"
                    << std::endl
                    << " Convergence rates for p = " << poly_degree
                    << std::endl
                    << " ********************************************"
                    << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("soln_Linf_error", true);
        convergence_table.set_scientific("output_error", true);
        if (this->pcout.is_active()) convergence_table.write_text(this->pcout.get_stream());

    }//end of grid loop
    
    return 0;//if reaches here mean passed test 
}

#if PHILIP_DIM==2
template class AdvectionPeriodic <PHILIP_DIM,1>;
#endif

} //Tests namespace
} //PHiLiP namespace
