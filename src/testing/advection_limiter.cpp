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
#include "advection_limiter.h"
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
        AdvectionLimiter<dim, nstate>::AdvectionLimiter(const PHiLiP::Parameters::AllParameters* const parameters_input)
            :
            TestsBase::TestsBase(parameters_input)
        {}


        template <int dim, int nstate>
        int AdvectionLimiter<dim, nstate>::run_test() const
        {
            pcout << " Running Advection limiter test. " << std::endl;
            pcout << dim << "    " << nstate << std::endl;
            PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
            double left = -1;
            double right = 1;
            const unsigned int n_grids = (!all_parameters_new.use_OOA) ? 5 : 8;
            unsigned int poly_degree = 3;
            const unsigned int igrid_start = (!all_parameters_new.use_OOA) ? 4 : 3;
            const unsigned int grid_degree = 1;
            dealii::ConvergenceTable convergence_table;
            std::vector<double> grid_size(n_grids);
            std::vector<double> soln_error(n_grids);

            for (unsigned int igrid = igrid_start; igrid < n_grids; igrid++) {

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
#if PHILIP_DIM==1
                std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
                dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
                grid->add_periodicity(matched_pairs);
#else
                std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
                dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
                if (dim >= 2) dealii::GridTools::collect_periodic_faces(*grid, 2, 3, 1, matched_pairs);
                if (dim >= 3) dealii::GridTools::collect_periodic_faces(*grid, 4, 5, 2, matched_pairs);
                grid->add_periodicity(matched_pairs);
#endif
                grid->refine_global(igrid);
                pcout << "Grid generated and refined" << std::endl;


                const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
                double n_dofs_cfl = pow(n_global_active_cells2, dim) * pow(poly_degree + 1.0, dim);
                double delta_x = (PHILIP_DIM == 2) ? (right - left) / pow(n_global_active_cells2, (1.0 / dim)) : (right - left) / pow(n_dofs_cfl, (1.0 / dim));
                all_parameters_new.ode_solver_param.initial_time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * pow(delta_x, (4.0/3.0)) : (1.0 / 3.0) * pow(delta_x, 2.0);
                pcout << "time_step:   " << all_parameters_new.ode_solver_param.initial_time_step << std::endl;

                //allocate dg
                std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
                pcout << "dg created" << std::endl;
                dg->allocate_system();

                //initialize IC
                pcout << "Setting up Initial Condition" << std::endl;
                //Create initial condition function
                std::shared_ptr< InitialConditionFunction<dim, nstate, double> > initial_condition_function =
                    InitialConditionFactory<dim, nstate, double>::create_InitialConditionFunction(&all_parameters_new);
                SetInitialCondition<dim, nstate, double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

                //get the global max and min of initial condition.
                dg->get_global_max_and_min_of_solution();
                // Create ODE solver using the factory and providing the DG object
                std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

                if (!all_parameters_new.use_OOA)
                {
                    double finalTime = 2.0;
                    ode_solver->advance_solution_time(finalTime);
                }
                else
                {
                    double finalTime = 0.1;

                    ode_solver->current_iteration = 0;

                    // advance solution until the final time
                    ode_solver->advance_solution_time(finalTime);

                    // output results
                    const unsigned int n_global_active_cells = grid->n_global_active_cells();
                    const unsigned int n_dofs = dg->dof_handler.n_dofs();
                    this->pcout << "Dimension: " << dim
                        << "\t Polynomial degree p: " << poly_degree
                        << std::endl
                        << "Grid number: " << igrid + 1 << "/" << n_grids
                        << ". Number of active cells: " << n_global_active_cells
                        << ". Number of degrees of freedom: " << n_dofs
                        << std::endl;

                    // Overintegrate the error to make sure there is not integration error in the error estimate
                    int overintegrate = 10;
                    dealii::QGauss<dim> quad_extra(poly_degree + 1 + overintegrate);
                    dealii::FEValues<dim, dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra,
                        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
                    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
                    std::array<double, nstate> soln_at_q;

                    double l2error = 0.0;

                    // Integrate every cell and compute L2
                    const double pi = atan(1) * 4.0;
                    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
                    const dealii::Tensor<1, 3, double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
                    for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell) {
                        if (!cell->is_locally_owned()) continue;

                        fe_values_extra.reinit(cell);
                        cell->get_dof_indices(dofs_indices);

                        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

                            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                            for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                            }

                            for (int istate = 0; istate < nstate; ++istate) {
                                const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                                double uexact = 1.0;
                                for (int idim = 0; idim < dim; idim++) {
                                    uexact *= sin(2.0 * pi * (qpoint[idim] - adv_speeds[idim] * finalTime));//for grid 1-3
                                }
                                l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                            }
                        }

                    }
                    const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, this->mpi_communicator));

                    // Convergence table
                    const double dx = 1.0 / pow(n_dofs, (1.0 / dim));
                    grid_size[igrid] = dx;
                    soln_error[igrid] = l2error_mpi_sum;

                    convergence_table.add_value("p", poly_degree);
                    convergence_table.add_value("cells", n_global_active_cells);
                    convergence_table.add_value("DoFs", n_dofs);
                    convergence_table.add_value("dx", dx);
                    convergence_table.add_value("soln_L2_error", l2error_mpi_sum);

                    this->pcout << " Grid size h: " << dx
                        << " L2-soln_error: " << l2error_mpi_sum
                        << " Residual: " << ode_solver->residual_norm
                        << std::endl;

                    if (igrid > igrid_start) {
                        const double slope_soln_err = log(soln_error[igrid] / soln_error[igrid - 1])
                            / log(grid_size[igrid] / grid_size[igrid - 1]);
                        this->pcout << "From grid " << igrid - 1
                            << "  to grid " << igrid
                            << "  dimension: " << dim
                            << "  polynomial degree p: " << poly_degree
                            << std::endl
                            << "  solution_error1 " << soln_error[igrid - 1]
                            << "  solution_error2 " << soln_error[igrid]
                            << "  slope " << slope_soln_err
                            << std::endl;
                    }

                    this->pcout << " ********************************************"
                        << std::endl
                        << " Convergence rates for p = " << poly_degree
                        << std::endl
                        << " ********************************************"
                        << std::endl;
                    convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
                    convergence_table.set_scientific("dx", true);
                    convergence_table.set_scientific("soln_L2_error", true);
                    if (this->pcout.is_active()) convergence_table.write_text(this->pcout.get_stream());
                }
            }//end of grid loop
            //want to add some condition to check for if(){return 1} else{return 0}
            return 0; //if got to here means passed the test, otherwise would've failed earlier
        }


        template class AdvectionLimiter <PHILIP_DIM, 1>;

    } //Tests namespace
} //PHiLiP namespace