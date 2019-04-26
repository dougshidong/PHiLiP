#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>


#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "physics/physics.h"
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

        Physics<dim,1,double> *physics_double = PhysicsFactory<dim, 1, double>::create_Physics(parameters.pde_type);

        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

            // p0 tends to require a finer grid to reach asymptotic region
            unsigned int n_grids = n_grids_input;
            if (poly_degree == 0) n_grids = n_grids_input + 2;

            std::vector<int> n_1d_cells(n_grids);
            n_1d_cells[0] = initial_grid_size;

            std::vector<double> soln_error(n_grids);
            std::vector<double> output_soln_error(n_grids);
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

                // Distort grid by random amount
                const double factor = 0.2; // should be less than 0.5
                const bool keep_boundary = true;
                //GridTools::distort_random (factor, grid, keep_boundary);

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

                // Overintegrate by alot
                QGauss<dim> quad_plus20(dg.fe.degree+10);
                FEValues<dim,dim> fe_values_plus20(dg.mapping, dg. fe,quad_plus20, update_values | update_JxW_values | update_quadrature_points);

                const unsigned int n_quad_pts = fe_values_plus20.n_quadrature_points;

                std::vector<double> solution_values_at_q(n_quad_pts);

                double l2error = 0;
                double exact_solution_integral = 0;
                double solution_integral = 0;
                typename DoFHandler<dim>::active_cell_iterator
                   cell = dg.dof_handler.begin_active(),
                   endc = dg.dof_handler.end();
                for (; cell!=endc; ++cell) {
                    //const unsigned int icell = cell->user_index();

                    fe_values_plus20.reinit (cell);
                    fe_values_plus20.get_function_values (dg.solution, solution_values_at_q);

                    double uexact = 0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        const Point<dim> qpoint = (fe_values_plus20.quadrature_point(iquad));

                        //uexact = manufactured_advection_solution (qpoint);
                        //uexact = manufactured_solution (qpoint);
                        physics_double->manufactured_solution (qpoint, uexact);


                        double u_at_q = solution_values_at_q[iquad];
                        l2error += pow(u_at_q - uexact, 2) * fe_values_plus20.JxW(iquad);

                        solution_integral += pow(u_at_q, 2) * fe_values_plus20.JxW(iquad);
                        exact_solution_integral += pow(uexact, 2) * fe_values_plus20.JxW(iquad);
                    }

                }
                const bool linear_output = false;
                exact_solution_integral = physics_double->integral_output(linear_output);
                l2error = sqrt(l2error);

                bool integrate_boundary = false;
                if (integrate_boundary) {
                    QGauss<dim-1> quad_face_plus20(dg.fe.degree+10);
                    exact_solution_integral = 0;
                    solution_integral = 0;
                    FEFaceValues<dim,dim> fe_face_values_plus20(dg.mapping, dg.fe, quad_face_plus20, update_normal_vectors | update_values | update_JxW_values | update_quadrature_points);
                    unsigned int n_face_quad_pts = fe_face_values_plus20.n_quadrature_points;
                    std::vector<double> face_intp_solution_values(n_face_quad_pts);

                    cell = dg.dof_handler.begin_active();
                    for (; cell!=endc; ++cell) {

                        //std::cout << "Cell loop" << std::endl;
                        for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
                            //std::cout << "Face loop" << std::endl;

                            typename DoFHandler<dim>::face_iterator current_face = cell->face(face_no);
                            fe_face_values_plus20.reinit (cell, face_no);
                            fe_face_values_plus20.get_function_values (dg.solution, face_intp_solution_values);

                            const std::vector<Tensor<1,dim> > &normals = fe_face_values_plus20.get_normal_vectors ();

                            if (current_face->at_boundary()) {
                                //std::cout << "Boundary if" << std::endl;

                                for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
                                    //std::cout << "Quad loop" << std::endl;
                                    const Point<dim> qpoint = (fe_face_values_plus20.quadrature_point(iquad));

                                    double uexact = 0;
                                    physics_double->manufactured_solution (qpoint, uexact);

                                    const Tensor<1,dim,double> characteristic_velocity_at_q = physics_double->convective_eigenvalues(uexact);
                                    const double vel_dot_normal = characteristic_velocity_at_q * normals[iquad];
                                    const bool inflow = (vel_dot_normal < 0.);
                                    if (inflow) {
                                    } else {
                                        double u_at_q = face_intp_solution_values[iquad];

                                        solution_integral += pow(u_at_q, 2) * fe_face_values_plus20.JxW(iquad);
                                        exact_solution_integral += pow(uexact, 2) * fe_face_values_plus20.JxW(iquad);
                                    }
                                }
                            }
                        }
                    }
                }


                double dx = 1.0/pow(n_active_cells,(1.0/dim));
                grid_size[igrid] = dx;
                soln_error[igrid] = l2error;
                output_soln_error[igrid] = std::abs(solution_integral - exact_solution_integral);


                std::cout   << " Grid size h: " << dx 
                            << " L2-soln_error: " << l2error
                            << " Residual: " << ode_solver->residual_norm
                            << std::endl;

                std::cout  
                            << " output_exact: " << exact_solution_integral
                            << " output_discrete: " << solution_integral
                            << " output_error: " << output_soln_error[igrid]
                            << std::endl;

                if (igrid > 0)
                std::cout << "From grid " << igrid-1
                          << "  to grid " << igrid
                          << "  dimension: " << dim
                          << "  polynomial degree p: " << dg.fe.get_degree()
                          << std::endl
                          << "  solution_error1 " << soln_error[igrid-1]
                          << "  solution_error2 " << soln_error[igrid]
                          << "  slope " << log((soln_error[igrid]/soln_error[igrid-1]))
                                          / log(grid_size[igrid]/grid_size[igrid-1])
                          << std::endl
                          << "  solution_integral_error1 " << output_soln_error[igrid-1]
                          << "  solution_integral_error2 " << output_soln_error[igrid]
                          << "  slope " << log((output_soln_error[igrid]/output_soln_error[igrid-1]))
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

                const double slope_soln_err = log(soln_error[igrid+1]/soln_error[igrid])
                                      / log(grid_size[igrid+1]/grid_size[igrid]);
                const double slope_output_err = log(output_soln_error[igrid+1]/output_soln_error[igrid])
                                      / log(grid_size[igrid+1]/grid_size[igrid]);
                std::cout << "From grid " << igrid+1
                          << "  to grid " << igrid+1+1
                          << "  dimension: " << dim
                          << "  polynomial degree p: " << poly_degree
                          << std::endl
                          << "  solution_error1 " << soln_error[igrid]
                          << "  solution_error2 " << soln_error[igrid+1]
                          << "  slope " << slope_soln_err
                          << std::endl
                          << "  solution_integral_error1 " << output_soln_error[igrid]
                          << "  solution_integral_error2 " << output_soln_error[igrid+1]
                          << "  slope " << slope_output_err
                          << std::endl;

            }
            std::cout << std::endl << std::endl;


            const double last_slope = log(soln_error[n_grids-1]/soln_error[n_grids-2])
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
