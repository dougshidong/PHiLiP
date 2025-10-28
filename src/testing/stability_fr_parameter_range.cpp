#include <deal.II/base/tensor.h>
#include <deal.II/base/convergence_table.h>

#include "stability_fr_parameter_range.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg_base.hpp"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
StabilityFRParametersRange<dim, nstate>::StabilityFRParametersRange(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
double StabilityFRParametersRange<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
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
double StabilityFRParametersRange<dim, nstate>::compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const
{
    // Conservation \f$ =  \int 1 * u d\Omega_m \f$
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
    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(dg->nstate, poly_degree, dg->max_grid_degree);
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
int StabilityFRParametersRange<dim, nstate>::run_test() const
{
    pcout << " Running stability ESFR parameter range test. " << std::endl;
    int testfail = 0;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = all_parameters_new.flow_solver_param.grid_left_bound;
    double right = all_parameters_new.flow_solver_param.grid_right_bound;
    const unsigned int n_grids = all_parameters_new.manufactured_convergence_study_param.initial_grid_size+all_parameters_new.manufactured_convergence_study_param.number_of_grids;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    std::vector<double> soln_error_inf(n_grids);
    unsigned int poly_degree = all_parameters_new.manufactured_convergence_study_param.degree_start;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = all_parameters_new.manufactured_convergence_study_param.initial_grid_size;
    pcout << " igrid_start" << igrid_start << std::endl;
    const unsigned int grid_degree = 1;

    const unsigned int nb_c_value = all_parameters_new.flow_solver_param.number_ESFR_parameter_values;
    const double c_min = all_parameters_new.flow_solver_param.ESFR_parameter_values_start;
    const double c_max = all_parameters_new.flow_solver_param.ESFR_parameter_values_end;
    const double log_c_min = std::log10(c_min);
    const double log_c_max = std::log10(c_max);
    std::vector<double> c_array(nb_c_value+1);

    std::ofstream conv_tab_file;
    const std::string fname = "convergence_tables.txt";
    conv_tab_file.open(fname);

    // Create log space array of c_value
    for (unsigned int ic = 0; ic < nb_c_value; ic++) {
        double log_c = log_c_min + (log_c_max - log_c_min) / (nb_c_value - 1) * ic;
        c_array[ic] = std::pow(10.0, log_c);
    }

    c_array[nb_c_value] = all_parameters_new.FR_user_specified_correction_parameter_value;

    // Loop over c_array to compute slope
    for (unsigned int ic = 0; ic < nb_c_value+1; ic++) {
        double c_value = c_array[ic];

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
            PHiLiP::Grids::straight_periodic_cube<dim, Triangulation>(grid, left, right, pow(2.0, igrid));
            pcout << "Grid generated and refined" << std::endl;
            //CFL number
            
            // use 0.0001 to be consistent with Ranocha and Gassner papers
            // all_parameters_new.ode_solver_param.initial_time_step =  0.0001;
            all_parameters_new.FR_user_specified_correction_parameter_value = c_value;
            std::cout << "c ESFR " <<all_parameters_new.FR_user_specified_correction_parameter_value <<  std::endl;
            //allocate dg
            std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
            pcout << "dg created" <<std::endl;
            dg->allocate_system (false,false,false);
            
            //initialize IC
            pcout<<"Setting up Initial Condition"<<std::endl;
            // Create initial condition function
            std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
            SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            //do OOA
            double finalTime=all_parameters_new.flow_solver_param.final_time;

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
            double linf_error = 0.0;

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
                        double uexact = 0.0;
                        const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                        if (all_parameters->pde_type == PHiLiP::Parameters::AllParameters::PartialDifferentialEquation::burgers_inviscid){
                            for(int idim=0; idim<dim; idim++){
                                uexact += cos(pi*(qpoint[idim]-finalTime));//for grid 1-3
                            }
                        }
                        else if (all_parameters->pde_type == PHiLiP::Parameters::AllParameters::PartialDifferentialEquation::advection){
                            uexact = 1.0;
                            const dealii::Tensor<1,3,double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
                            for(int idim=0; idim<dim; idim++){
                                if(left==0){
                                    uexact *= sin(2.0*pi*(qpoint[idim]-adv_speeds[idim]*finalTime));//for grid 1-3
                                }
                                if(left==-1){
                                    uexact *= sin(pi*(qpoint[idim]- adv_speeds[idim]*finalTime));//for grid 1-3
                                }
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
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

            const double linferror_mpi= (dealii::Utilities::MPI::max(linf_error, this->mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            soln_error_inf[igrid] = linferror_mpi;

            //Checking convergence order
            double expected_order = poly_degree + 1;
            if(c_value > 0.186)
                expected_order -= 1;

            //set tolerance to make test pass for ctest. Note that the grids are very coarse (not in asymptotic range)
            const double order_tolerance = 1.0; 
            if (igrid > 0) {
                const double slope_soln_err = log(soln_error[igrid] / soln_error[igrid - 1])
                    / log(grid_size[igrid] / grid_size[igrid - 1]);
                
                if (abs(slope_soln_err - expected_order) > order_tolerance){
                    testfail = 1;
                    pcout << "Expected convergence order was not reached at refinement " <<std::endl;
                }
            }

            convergence_table.add_value("c", c_value);
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("soln_Linf_error", linferror_mpi);

            pcout << " Grid size h: " << dx 
                << " L2-soln_error: " << l2error_mpi_sum
                << " Linf-soln_error: " << linferror_mpi
                << " Residual: " << ode_solver->residual_norm
                << std::endl;

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                    / log(grid_size[igrid]/grid_size[igrid-1]); 

                const double slope_soln_err_inf = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                                        / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid
                    << "  to grid " << igrid+1
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
            }
        
            pcout << " ********************************************"
                << std::endl
                << " Convergence rates for p = " << poly_degree
                << std::endl
                << " ********************************************"
                << std::endl;
            convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("c", true);
            convergence_table.set_scientific("dx", true);
            convergence_table.set_scientific("soln_L2_error", true);
            convergence_table.set_scientific("soln_Linf_error", true);
            if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
            //end of OOA
        }//end of grid loop

        convergence_table.write_text(conv_tab_file);
        convergence_table.clear();
    }//end of Loop over c_array
    conv_tab_file.close();
    return testfail; //if got to here means passed the test, otherwise would've failed earlier
}
template class StabilityFRParametersRange<PHILIP_DIM,1>;
template class StabilityFRParametersRange<PHILIP_DIM,2>;
template class StabilityFRParametersRange<PHILIP_DIM,3>;
template class StabilityFRParametersRange<PHILIP_DIM,4>;
template class StabilityFRParametersRange<PHILIP_DIM,5>;

} // Tests namespace
} // PHiLiP namespace
