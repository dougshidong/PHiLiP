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

namespace PHiLiP
{
    using namespace dealii;

    template <int dim>
    Point<dim> warp (const Point<dim> &p)
    {
      Point<dim> q = p;
      q[dim-1] *= 1.5;
      if (dim >= 2)
        q[0] += 1*std::sin(q[dim-1]);
      if (dim >= 3)
        q[1] += 1*std::cos(q[dim-1]);
      return q;
    }

    template <int dim>
    void print_mesh_info(const Triangulation<dim> &triangulation,
                         const std::string        &filename)
    {
      std::cout << "Mesh info:" << std::endl
                << " dimension: " << dim << std::endl
                << " no. of cells: " << triangulation.n_active_cells() << std::endl;
      {
        std::map<types::boundary_id, unsigned int> boundary_count;
        for (auto cell : triangulation.active_cell_iterators())
          {
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary())
                  boundary_count[cell->face(face)->boundary_id()]++;
              }
          }
        std::cout << " boundary indicators: ";
        for (const std::pair<const types::boundary_id, unsigned int> &pair : boundary_count)
          {
            std::cout << pair.first << "(" << pair.second << " times) ";
          }
        std::cout << std::endl;
      }
      std::ofstream out (filename);
      GridOut grid_out;
      grid_out.write_eps (triangulation, out);
      std::cout << " written to " << filename
                << std::endl
                << std::endl;
    }

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

        std::vector<int> fail_conv_poly;
        std::vector<double> fail_conv_slop;
        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

            // p0 tends to require a finer grid to reach asymptotic region
            unsigned int n_grids = n_grids_input;
            if (poly_degree == 0) n_grids = n_grids_input + 2;

            std::vector<int> n_1d_cells(n_grids);
            n_1d_cells[0] = initial_grid_size;

            std::vector<double> soln_error(n_grids);
            std::vector<double> output_error(n_grids);
            std::vector<double> grid_size(n_grids);

            for (unsigned int i=1;i<n_grids;++i) {
                n_1d_cells[i] = n_1d_cells[i-1]*grid_progression;
            }

            ConvergenceTable convergence_table;
            for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
                // Note that Triangulation must be declared before DiscontinuousGalerkin
                // DiscontinuousGalerkin will be destructed before Triangulation
                // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
                // Otherwise, a Subscriptor error will occur

                Triangulation<dim> grid;

                bool generate_square_mesh = true;
                bool sine_mesh = false;
                const double factor = 0.0000; // should be less than 0.5
                const bool keep_boundary = true;
                bool readmesh = false, writemesh = true;

                if (generate_square_mesh) {
                    GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid]);
                    for (
                        typename Triangulation<dim>::active_cell_iterator
                        cell = grid.begin_active(); cell != grid.end(); ++cell)
                    {
                        cell->set_material_id(9002);
                        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                            if (cell->face(f)->at_boundary())
                                  cell->face(f)->set_boundary_id (9001);
                    }
                }
                //if (generate_square_mesh) GridGenerator::hyper_cube_with_cylindrical_hole(grid, 0.25, 0.5, 0.5, n_1d_cells[igrid],false);

                //std::string read_mshname = "squareunsquad0.msh";
                //std::cout<<"Reading grid: " << read_mshname << std::endl;
                //std::ifstream inmesh(read_mshname);
                //GridIn<dim,dim> grid_in;
                //grid_in.attach_triangulation(grid);
                //grid_in.read_msh(inmesh);
                //grid.refine_global(igrid);


                //   Distort grid by random amount
                if (factor >= 1e-10) GridTools::distort_random (factor, grid, keep_boundary);
                //Point<dim> zero;
                //GridTools::rotate (30, grid);
                if (sine_mesh) GridTools::transform (&warp<dim>, grid);


                if (readmesh) {
                    //std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                    std::string read_mshname = "squareunsquad"+std::to_string(igrid)+".msh";
                    std::cout<<"Reading grid: " << read_mshname << std::endl;
                    std::ifstream inmesh(read_mshname);
                    GridIn<dim,dim> grid_in;
                    grid_in.attach_triangulation(grid);
                    grid_in.read_msh(inmesh);
                }
                if (writemesh) {
                    std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                    std::ofstream outmesh(write_mshname);
                    GridOutFlags::Msh msh_flags(true, true);
                    GridOut grid_out;
                    grid_out.set_flags(msh_flags);
                    grid_out.write_msh(grid, outmesh);
                }


                std::string gridname = "grid-"+std::to_string(igrid)+".eps";
                if (dim == 2) print_mesh_info (grid, gridname);


                const int nstate = 1;
                using ADtype = Sacado::Fad::DFad<double>;
                Physics<dim, nstate, ADtype> *physics = PhysicsFactory<dim, nstate, ADtype >::create_Physics(parameters.pde_type);

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
                          << "Grid number: " << igrid+1 << "/" << n_grids
                          << ". Number of active cells: " << n_active_cells
                          << ". Number of degrees of freedom: " << dg.dof_handler.n_dofs()
                          << std::endl;

                ode_solver->steady_state();
                dg.output_results(igrid);

                std::vector<unsigned int> dof_indices(dg.fe.dofs_per_cell);

                // Overintegrate by alot
                QGauss<dim> quad_plus20(dg.fe.degree+5);
                FEValues<dim,dim> fe_values_plus20(dg.mapping, dg.fe, quad_plus20, update_values | update_JxW_values | update_quadrature_points);

                const unsigned int n_quad_pts = fe_values_plus20.n_quadrature_points;

                std::vector<double> solution_values_at_q(n_quad_pts);

                double l2error = 0;

                bool linear_output = false;
                //linear_output = true;
                int power;
                if (linear_output) power = 1;
                if (!linear_output) power = 2;

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

                        solution_integral += pow(u_at_q, power) * fe_values_plus20.JxW(iquad);
                    }

                }
                const double exact_solution_integral = physics_double->integral_output(linear_output);
                l2error = sqrt(l2error);

                bool integrate_boundary = false;
                if (integrate_boundary) {
                    QGauss<dim-1> quad_face_plus20(dg.fe.degree+5);
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

                                    std::vector<double> characteristic_dot_n_at_q = physics_double->convective_eigenvalues(uexact, normals[iquad]);
                                    const bool inflow = (characteristic_dot_n_at_q[0] < 0.);
                                    if (inflow) {
                                    } else {
                                        double u_at_q = face_intp_solution_values[iquad];

                                        solution_integral += pow(u_at_q, 2) * fe_face_values_plus20.JxW(iquad);
                                    }
                                }
                            }
                        }
                    }
                }


                double dx = 1.0/pow(n_active_cells,(1.0/dim));
                dx = GridTools::maximal_cell_diameter(grid);
                grid_size[igrid] = dx;
                soln_error[igrid] = l2error;
                output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

                convergence_table.add_value("cells", grid.n_active_cells());
                convergence_table.add_value("dx", dx);
                convergence_table.add_value("soln_L2_error", l2error);
                convergence_table.add_value("output_error", output_error[igrid]);



                std::cout   << " Grid size h: " << dx 
                            << " L2-soln_error: " << l2error
                            << " Residual: " << ode_solver->residual_norm
                            << std::endl;

                std::cout  
                            << " output_exact: " << exact_solution_integral
                            << " output_discrete: " << solution_integral
                            << " output_error: " << output_error[igrid]
                            << std::endl;

                if (igrid > 0) {
                    const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                          / log(grid_size[igrid]/grid_size[igrid-1]);
                    const double slope_output_err = log(output_error[igrid]/output_error[igrid-1])
                                          / log(grid_size[igrid]/grid_size[igrid-1]);
                    std::cout << "From grid " << igrid-1
                              << "  to grid " << igrid
                              << "  dimension: " << dim
                              << "  polynomial degree p: " << dg.fe.get_degree()
                              << std::endl
                              << "  solution_error1 " << soln_error[igrid-1]
                              << "  solution_error2 " << soln_error[igrid]
                              << "  slope " << slope_soln_err
                              << std::endl
                              << "  solution_integral_error1 " << output_error[igrid-1]
                              << "  solution_integral_error2 " << output_error[igrid]
                              << "  slope " << slope_output_err
                              << std::endl;
                }

                //output_results (igrid);
                delete ode_solver;
            }
            std::cout
                << " ********************************************"
                << std::endl
                << " Convergence rates for p = " << poly_degree
                << std::endl
                << " ********************************************"
                << std::endl;
            convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.evaluate_convergence_rates("output_error", "cells", ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("dx", true);
            convergence_table.set_scientific("soln_L2_error", true);
            convergence_table.set_scientific("output_error", true);
            convergence_table.write_text(std::cout);

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
                fail_conv_poly.push_back(poly_degree);
                fail_conv_slop.push_back(last_slope);
            }

        }
        int n_fail_poly = fail_conv_poly.size();
        if (n_fail_poly > 0) {
            for (int ifail=0; ifail < n_fail_poly; ++ifail) {
                const double expected_slope = fail_conv_poly[ifail]+1;
                const double slope_deficit_tolerance = -0.1;
                std::cout << std::endl
                          << "Convergence order not achieved for polynomial p = "
                          << fail_conv_poly[ifail]
                          << ". Slope of "
                          << fail_conv_slop[ifail] << " instead of expected "
                          << expected_slope << " within a tolerance of "
                          << slope_deficit_tolerance
                          << std::endl;
            }
        }
        return n_fail_poly;

    }
    template int manufactured_grid_convergence<PHILIP_DIM> (Parameters::AllParameters &parameters);

} // end of PHiLiP namespace
