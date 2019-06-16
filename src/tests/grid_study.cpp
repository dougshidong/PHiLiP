#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <Sacado.hpp>

#include "tests.h"
#include "grid_study.h"

#include "physics/physics.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

//#include "template_instantiator.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
GridStudy<dim,nstate>::GridStudy(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
void GridStudy<dim,nstate>
::initialize_perturbed_solution(DGBase<dim,double> &dg, const Physics::PhysicsBase<dim,nstate,double> &physics) const
{
    dealii::VectorTools::interpolate(dg.dof_handler, physics.manufactured_solution_function, dg.solution);
}
template <int dim, int nstate>
double GridStudy<dim,nstate>
::integrate_solution_over_domain(DGBase<dim,double> &dg) const
{
    std::cout << "Evaluating solution integral..." << std::endl;
    double solution_integral = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.fe_system.tensor_degree()+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(dg.mapping, dg.fe_system, quad_extra, 
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    const bool linear_output = false;
    int power;
    if (linear_output) power = 1;
    else power = 2;

    // Integrate solution error and output error
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Interpolate solution to quadrature points
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }
            // Integrate solution
            for (int s=0; s<nstate; s++) {
                solution_integral += pow(soln_at_q[0], power) * fe_values_extra.JxW(iquad);
            }
        }

    }
    return solution_integral;
}

template<int dim, int nstate>
int GridStudy<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int initial_grid_size   = manu_grid_conv_param.initial_grid_size;
    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;
    const double       grid_progression    = manu_grid_conv_param.grid_progression;



    Physics::PhysicsBase<dim,nstate,double> *physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(param.pde_type);

    // Evaluate solution integral on really fine mesh
    double exact_solution_integral;
    std::cout << "Evaluating EXACT solution integral..." << std::endl;
    // Limit the scope of grid_super_fine and dg_super_fine
    {
        dealii::Triangulation<dim> grid_super_fine;
        dealii::GridGenerator::subdivided_hyper_cube(grid_super_fine, initial_grid_size*pow(grid_progression,n_grids_input));
        std::shared_ptr < DGBase<dim, double> > dg_super_fine = DGFactory<dim,double>::create_discontinuous_galerkin(&param, p_end);
        dg_super_fine->set_triangulation(&grid_super_fine);
        dg_super_fine->allocate_system ();

        initialize_perturbed_solution(*dg_super_fine, *physics_double);
        exact_solution_integral = integrate_solution_over_domain(*dg_super_fine);
        std::cout << "Exact solution integral is " << exact_solution_integral << std::endl;
    }

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input + 2;

        std::vector<int> n_1d_cells(n_grids);
        n_1d_cells[0] = initial_grid_size;
        if(poly_degree==0) n_1d_cells[0] = initial_grid_size + 1;

        std::vector<double> soln_error(n_grids);
        std::vector<double> output_error(n_grids);
        std::vector<double> grid_size(n_grids);

        for (unsigned int igrid=1;igrid<n_grids;++igrid) {
            n_1d_cells[igrid] = n_1d_cells[igrid-1]*grid_progression;
        }

        dealii::ConvergenceTable convergence_table;

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
            // Note that Triangulation must be declared before DG
            // DG will be destructed before Triangulation
            // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
            // Otherwise, a Subscriptor error will occur
            dealii::Triangulation<dim> grid;

            // Generate hypercube
            if (   manu_grid_conv_param.grid_type == GridEnum::hypercube
                || manu_grid_conv_param.grid_type == GridEnum::sinehypercube ) {
                dealii::GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid]);
                for (typename dealii::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
                    // Set a dummy boundary ID
                    cell->set_material_id(9002);
                    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                        if (cell->face(face)->at_boundary())
                              cell->face(face)->set_boundary_id (9001);
                }
                // Warp grid if requested in input file
                if (manu_grid_conv_param.grid_type == GridEnum::sinehypercube) dealii::GridTools::transform (&warp, grid);
            }

            // Distort grid by random amount if requested
            const double random_factor = manu_grid_conv_param.random_distortion;
            const bool keep_boundary = true;
            if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, grid, keep_boundary);

            // Read grid if requested
            if (manu_grid_conv_param.grid_type == GridEnum::read_grid) {
                //std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::string read_mshname = manu_grid_conv_param.input_grids+std::to_string(igrid)+".msh";
                std::cout<<"Reading grid: " << read_mshname << std::endl;
                std::ifstream inmesh(read_mshname);
                dealii::GridIn<dim,dim> grid_in;
                grid_in.attach_triangulation(grid);
                grid_in.read_msh(inmesh);
            }
            // Output grid if requested
            if (manu_grid_conv_param.output_meshes) {
                std::string write_mshname = "grid-"+std::to_string(igrid)+".msh";
                std::ofstream outmesh(write_mshname);
                dealii::GridOutFlags::Msh msh_flags(true, true);
                dealii::GridOut grid_out;
                grid_out.set_flags(msh_flags);
                grid_out.write_msh(grid, outmesh);
            }

            // Show mesh if in 2D
            std::string gridname = "grid-"+std::to_string(igrid)+".eps";
            if (dim == 2) print_mesh_info (grid, gridname);

            using ADtype = Sacado::Fad::DFad<double>;

            // Create DG object using the factory
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree);
            dg->set_triangulation(&grid);
            dg->allocate_system ();
            //dg->evaluate_inverse_mass_matrices();
            //
            // PhysicsBase required for exact solution and output error
            std::cout<<"Test Physics nstate" << nstate << std::endl;

            initialize_perturbed_solution(*(dg), *(physics_double));

            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            unsigned int n_active_cells = grid.n_active_cells();
            std::cout
                      << "Dimension: " << dim
                      << "\t Polynomial degree p: " << poly_degree
                      << std::endl
                      << "Grid number: " << igrid+1 << "/" << n_grids
                      << ". Number of active cells: " << n_active_cells
                      << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
                      << std::endl;

            // Solve the steady state problem
            ode_solver->steady_state();

            // Output the solution to gnuplot. Can only visualize 1D for now
            if(dim==1) dg->output_results(igrid);

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->fe_system.tensor_degree()+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(dg->mapping, dg->fe_system, quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0;

            // Integrate solution error and output error
            typename dealii::DoFHandler<dim>::active_cell_iterator
               cell = dg->dof_handler.begin_active(),
               endc = dg->dof_handler.end();

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for (; cell!=endc; ++cell) {

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    //std::array<double,nstate> uexact;
                    //std::cout << "cos(0.59*x+1 " << cos(0.59*qpoint[0]+1) << std::endl;
                    //std::cout << "uexact[1] " << uexact[1] << std::endl;

                    for (int istate=0; istate<nstate; ++istate) {
                        const double uexact = physics_double->manufactured_solution_function.value(qpoint, istate);
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                    // Only integrate first state variable for output error
                }

            }
            l2error = sqrt(l2error);

            double solution_integral = integrate_solution_over_domain(*dg);

            //  // Integrate boundary. Not needed for now. Might need something like this for adjoint consistency later on
            //  bool integrate_boundary = false;
            //  if (integrate_boundary) {
            //      QGauss<dim-1> quad_face_plus20(dg->fe_system.tensor_degree()+overintegrate);
            //      solution_integral = 0;
            //      FEFaceValues<dim,dim> fe_face_values_plus20(dg->mapping, dg->fe_system, quad_face_plus20, update_normal_vectors | update_values | update_JxW_values | update_quadrature_points);
            //      unsigned int n_face_quad_pts = fe_face_values_plus20.n_quadrature_points;
            //      std::vector<double> face_intp_solution_values(n_face_quad_pts);

            //      cell = dg->dof_handler.begin_active();
            //      for (; cell!=endc; ++cell) {

            //          for (unsigned int face_no=0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {

            //              typename dealii::DoFHandler<dim>::face_iterator current_face = cell->face(face_no);
            //              fe_face_values_plus20.reinit (cell, face_no);
            //              fe_face_values_plus20.get_function_values (dg->solution, face_intp_solution_values);

            //              const std::vector<Tensor<1,dim> > &normals = fe_face_values_plus20.get_normal_vectors ();

            //              if (current_face->at_boundary()) {

            //                  for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            //                      const dealii::Point<dim> qpoint = (fe_face_values_plus20.quadrature_point(iquad));

            //                      std::array<double,nstate> uexact;
            //                      physics_double->manufactured_solution (qpoint, uexact);

            //                      std::array<double,nstate> characteristic_dot_n_at_q = physics_double->convective_eigenvalues(uexact, normals[iquad]);
            //                      const int istate = 0;
            //                      const bool inflow = (characteristic_dot_n_at_q[istate] < 0.);
            //                      if (inflow) {
            //                      } else {
            //                          double u_at_q = face_intp_solution_values[iquad];

            //                          solution_integral += pow(u_at_q, 2) * fe_face_values_plus20.JxW(iquad);
            //                      }
            //                  }
            //              }
            //          }
            //      }
            //  }

            // Convergence table
            double dx = 1.0/pow(n_active_cells,(1.0/dim));
            dx = dealii::GridTools::maximal_cell_diameter(grid);
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error;
            output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

            convergence_table.add_value("p", poly_degree);
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
                          << "  polynomial degree p: " << dg->fe_system.tensor_degree()
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
        }
        std::cout
            << " ********************************************"
            << std::endl
            << " Convergence rates for p = " << poly_degree
            << std::endl
            << " ********************************************"
            << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("output_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("output_error", true);
        convergence_table.write_text(std::cout);

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(soln_error[n_grids-1]/soln_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        double before_last_slope = last_slope;
        if ( n_grids > 2 ) {
        before_last_slope = log(soln_error[n_grids-2]/soln_error[n_grids-3])
                            / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        }
        const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -0.1;
        if(poly_degree == 0) slope_deficit_tolerance = -0.2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            std::cout << std::endl
                      << "Convergence order not achieved. Average last 2 slopes of "
                      << slope_avg << " instead of expected "
                      << expected_slope << " within a tolerance of "
                      << slope_deficit_tolerance
                      << std::endl;
            fail_conv_poly.push_back(poly_degree);
            fail_conv_slop.push_back(slope_avg);
        }

    }
    std::cout << std::endl
              << std::endl
              << std::endl
              << std::endl;
    std::cout << " ********************************************"
              << std::endl;
    std::cout << " Convergence summary"
              << std::endl;
    std::cout << " ********************************************"
              << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        conv->write_text(std::cout);
        std::cout << " ********************************************"
                  << std::endl;
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

template <int dim, int nstate>
dealii::Point<dim> GridStudy<dim,nstate>
::warp (const dealii::Point<dim> &p)
{
    dealii::Point<dim> q = p;
    q[dim-1] *= 1.5;
    if (dim >= 2) q[0] += 1*std::sin(q[dim-1]);
    if (dim >= 3) q[1] += 1*std::cos(q[dim-1]);
    return q;
}

template <int dim, int nstate>
void GridStudy<dim,nstate>
::print_mesh_info(const dealii::Triangulation<dim> &triangulation, const std::string &filename) const
{
    std::cout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << triangulation.n_active_cells() << std::endl;
    {
        std::map<dealii::types::boundary_id, unsigned int> boundary_count;
        for (auto cell : triangulation.active_cell_iterators()) {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) boundary_count[cell->face(face)->boundary_id()]++;
            }
        }
        std::cout << " boundary indicators: ";
        for (const std::pair<const dealii::types::boundary_id, unsigned int> &pair : boundary_count) {
            std::cout << pair.first << "(" << pair.second << " times) ";
        }
        std::cout << std::endl;
    }
    if (dim == 2) {
        std::ofstream out (filename);
        dealii::GridOut grid_out;
        grid_out.write_eps (triangulation, out);
        std::cout << " written to " << filename << std::endl << std::endl;
    }
}

template class GridStudy <PHILIP_DIM,1>;
template class GridStudy <PHILIP_DIM,2>;
template class GridStudy <PHILIP_DIM,3>;
template class GridStudy <PHILIP_DIM,4>;
template class GridStudy <PHILIP_DIM,5>;
//template struct Instantiator<GridStudy,3,5>;



} // Tests namespace
} // PHiLiP namespace
