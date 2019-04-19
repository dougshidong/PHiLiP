#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "physics.h"
#include "dg.h"
#include "linear_solver.h"
#include "ode_solver.h"

#include "manufactured_solution.h"
namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    int manufactured_grid_convergence (Parameters::AllParameters &parameters)
    {
        Assert(dim == parameters.dimension, ExcDimensionMismatch(dim, parameters.dimension));

        const unsigned int p_start             = parameters.degree_start;
        const unsigned int p_end               = parameters.degree_end;

        const unsigned int initial_grid_size   = parameters.initial_grid_size;
        const unsigned int n_grids_input       = parameters.number_of_grids;
        const double       grid_progression    = parameters.grid_progression;

        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

            // p0 tends to require a finer grid to reach asymptotic region
            unsigned int n_grids = n_grids_input;
            if (poly_degree == 0) n_grids = n_grids_input + 2;

            std::vector<int> n_1d_cells(n_grids);
            n_1d_cells[0] = initial_grid_size;

            std::vector<double> error(n_grids);
            std::vector<double> grid_size(n_grids);

            for (unsigned int i=1;i<n_grids;++i) {
                n_1d_cells[i] = n_1d_cells[i-1]*grid_progression;
            }

            for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
                // Note that Triangulation must be declared before DiscontinuousGalerkin
                // DiscontinuousGalerkin will be destructed before Triangulation
                // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
                // Otherwise, a Subscriptor error will occur
                Triangulation<dim> grid;
                std::cout << "Generating hypercube for grid convergence... " << std::endl;
                GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid]);

                Physics<dim,1,Sacado::Fad::DFad<double>> *physics = PhysicsFactory<dim, 1, Sacado::Fad::DFad<double> >::create_Physics(parameters.pde_type);

                DiscontinuousGalerkin<PHILIP_DIM, double> dg(&parameters, poly_degree);
                dg.set_triangulation(&grid);
                dg.set_physics(physics);
                dg.allocate_system ();

                //ODESolver<dim, double> *ode_solver = ODESolverFactory<dim, double>::create_ODESolver(parameters.solver_type);
                ODESolver<dim, double> *ode_solver = ODESolverFactory<dim, double>::create_ODESolver(&dg);

                unsigned int n_active_cells = grid.n_active_cells();
                std::cout
                          << "Dimension: " << dim
                          << "\t Polynomial degree p: " << poly_degree
                          << std::endl
                          << "Grid number: " << igrid+1 << "/" << n_grids+1
                          << ". Number of active cells: " << n_active_cells
                          << ". Number of degrees of freedom: " << dg.dof_handler.n_dofs()
                          << std::endl;

                ode_solver->steady_state();
                dg.output_results(igrid);

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

                        //uexact = manufactured_advection_solution (qpoint);
                        uexact = manufactured_solution (qpoint);


                        double u_at_q = solution_values[iquad];
                        l2error += pow(u_at_q - uexact, 2) * fe_values_plus10.JxW(iquad);
                    }

                }
                l2error = sqrt(l2error);


                double dx = 1.0/pow(n_active_cells,(1.0/dim));
                grid_size[igrid] = dx;
                error[igrid] = l2error;


                std::cout   << " Grid size h: " << dx 
                            << " L2-error: " << l2error
                            << " Residual: " << ode_solver->residual_norm
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
                          << "From grid " << igrid+1
                          << "  to grid " << igrid+1+1
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
            const double slope_diff = last_slope-expected_slope;
            const double slope_deficit_tolerance = -0.1;

            if (slope_diff < slope_deficit_tolerance) {
                std::cout << std::endl
                          << "Convergence order not achieved. Slope of "
                          << last_slope << " instead of expected "
                          << expected_slope << " within a tolerance of "
                          << slope_deficit_tolerance
                          << std::endl;
                return 1;
            }

        }
        return 0;

    }
    template int manufactured_grid_convergence<PHILIP_DIM> (Parameters::AllParameters &parameters);

} // end of PHiLiP namespace
