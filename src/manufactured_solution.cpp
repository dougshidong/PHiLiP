#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_values.h>

#include "dg.h"
#include "linear_solver.h"
#include "ode_solver.h"
namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    int manufactured_grid_convergence (Parameters::AllParameters &parameters)
    {
        Assert(dim == parameters.dimension, ExcDimensionMismatch(dim, parameters.dimension));
        const unsigned int p_start = parameters.degree_start;
        const unsigned int p_end = parameters.degree_end;
        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {


            unsigned int n_grids = 5;
            std::vector<double> error(n_grids);
            std::vector<double> grid_size(n_grids);
            std::vector<double> ncell(n_grids);

            ncell[0] = 2;
            ncell[1] = 4;
            ncell[2] = 6;
            ncell[3] = 8;
            ncell[4] = 10;

            ncell[0] = 2;
            for (unsigned int i=1;i<n_grids;++i) {
                ncell[i] = ncell[i-1]*1.5;
            }


            for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


                // Note that Triangulation must be declared before DiscontinuousGalerkin
                // DiscontinuousGalerkin will be destructed before Triangulation
                // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
                // Otherwise, a Subscriptor error will occur
                Triangulation<dim> grid;
                DiscontinuousGalerkin<PHILIP_DIM, double> dg(&parameters, poly_degree);

                //ODESolver<dim, double> *ode_solver = ODESolverFactory<dim, double>::create_ODESolver(parameters.solver_type);
                ODESolver<dim, double> *ode_solver = ODESolverFactory<dim, double>::create_ODESolver(&dg);


                std::cout << "Generating hypercube for grid convergence... " << std::endl;
                GridGenerator::subdivided_hyper_cube(grid, ncell[igrid]);

                dg.set_triangulation(&grid);

                
                dg.allocate_system ();


                unsigned int n_active_cells = grid.n_active_cells();
                std::cout << "Grid number: " << igrid 
                          << ". Number of active cells: " << n_active_cells
                          << ". Number of degrees of freedom: " << dg.dof_handler.n_dofs()
                          << std::endl;

                ode_solver->steady_state();

                //double residual_norm = dg.get_residual_l2norm();


                //int iteration = 0;
                //int print = 1;

                //dg.assemble_system ();
                //while (residual_norm > 1e-13 && iteration < 100000) {
                //    ++iteration;


                //    dg.right_hand_side = 0;
                //    dg.assemble_system ();
                //    residual_norm = dg.get_residual_l2norm();

                //    if ( (iteration%print) == 0)
                //    std::cout << " Iteration: " << iteration 
                //              << " Residual norm: " << residual_norm
                //              << std::endl;

                //    dg.newton_update = 0;
                //    std::pair<unsigned int, double> convergence = solve_linear (
                //        dg.system_matrix,
                //        dg.right_hand_side, 
                //        dg.newton_update);

                //    std::cout << " Iteration: " << iteration 
                //              << " Newton update norm: " << dg.newton_update.l2_norm()
                //              << std::endl;

                //    dg.solution += dg.newton_update;
                //}

                //dg.delete_fe_values ();

                std::vector<unsigned int> dof_indices(dg.fe.dofs_per_cell);

                QGauss<dim> quad_plus10(dg.fe.degree+10);
                const unsigned int n_quad_pts = quad_plus10.size();
                FEValues<dim,dim> fe_values_plus10(dg.mapping, dg.fe,quad_plus10, update_values | update_JxW_values | update_quadrature_points);

                std::vector<double> solution_values(n_quad_pts);

                double l2error = 0;
                typename DoFHandler<dim>::active_cell_iterator
                   cell = dg.dof_handler.begin_active(),
                   endc = dg.dof_handler.end();
                for (; cell!=endc; ++cell) {
                    //const unsigned int icell = cell->user_index();

                    fe_values_plus10.reinit (cell);
                    fe_values_plus10.get_function_values (dg.solution, solution_values);

                    double uexact = 0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        const Point<dim> qpoint = (fe_values_plus10.quadrature_point(iquad));
                        if (dim==1) uexact = sin(3.19/dim*qpoint(0));
                        if (dim==2) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1));
                        if (dim==3) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1))*sin(3.19/dim*qpoint(2));

                        double u_at_q = solution_values[iquad];
                        l2error += pow(u_at_q - uexact, 2) * fe_values_plus10.JxW(iquad);
                    }

                }
                l2error = sqrt(l2error);


                double dx = 1.0/pow(n_active_cells,(1.0/dim));
                grid_size[igrid] = dx;
                error[igrid] = l2error;


                std::cout   << " dx: " << dx 
                            << " l2error: " << l2error
                            << " residual: " << ode_solver->residual_norm
                            << std::endl;

                if (igrid > 0)
                std::cout << "From grid " << igrid-1
                          << "  to grid " << igrid
                          << "  e1 " << error[igrid-1]
                          << "  e2 " << error[igrid]
                          << "  dimension: " << dim
                          << "  polynomial degree p: " << dg.fe.get_degree()
                          << "  slope " << log((error[igrid]/error[igrid-1]))
                                          / log(grid_size[igrid]/grid_size[igrid-1])
                          << std::endl;

                //output_results (igrid);
                delete ode_solver;
            }
            //dg.triangulation->list_subscribers();
            //grid->list_subscribers();
            //std::cout<<std::flush;
            //deallog<<std::flush;

            // CURRENTLY MEMORY LEAK
            // NEED TO DEALLOCATE THE GRID
            // DON'T KNOW HOW TO DEALLOCATE DUE TO SUBSCRIPTOR THING
            //delete dg.triangulation;
            //delete grid;
            //grid = NULL;

            std::cout << std::endl << std::endl;
            for (unsigned int igrid=0; igrid<n_grids-1; ++igrid) {

                const double slope = log(error[igrid+1]/error[igrid])
                                      / log(grid_size[igrid+1]/grid_size[igrid]);
                std::cout
                          << "From grid " << igrid
                          << "  to grid " << igrid+1
                          << "  e1 " << error[igrid]
                          << "  e2 " << error[igrid+1]
                          << "  dimension: " << dim
                          << "  polynomial degree p: " << poly_degree
                          << "  slope " << slope
                          << std::endl;

            }
            std::cout << std::endl << std::endl;


            const double last_slope = log(error[n_grids-1]/error[n_grids-2])
                                      / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
            const double expected_slope = poly_degree+1;
            const double slope_diff = std::abs(last_slope-expected_slope);
            const double slope_tolerance = 0.1;

            if (slope_diff > slope_tolerance) {
                std::cout << std::endl
                          << "Convergence order not achieved. Slope of "
                          << last_slope << " instead of expected "
                          << expected_slope << " within a tolerance of "
                          << slope_tolerance
                          << std::endl;
                return 1;
            }

        }
        return 0;

    }
    template int manufactured_grid_convergence<PHILIP_DIM> (Parameters::AllParameters &parameters);

} // end of PHiLiP namespace
