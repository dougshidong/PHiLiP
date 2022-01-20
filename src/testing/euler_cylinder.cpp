#include <stdlib.h>
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>


#include "euler_cylinder.h"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"


namespace PHiLiP {
namespace Tests {

dealii::Point<2> center(0.0,0.0);
const double inner_radius = 1, outer_radius = inner_radius*20;

dealii::Point<2> warp_cylinder (const dealii::Point<2> &p)//, const double inner_radius, const double outer_radius)
{
    const double rectangle_height = 1.0;
    //const double original_radius = std::abs(p[0]);
    const double angle = p[1]/rectangle_height * dealii::numbers::PI;

    //const double radius = std::abs(p[0]);

    const double power = 1.8;
    const double radius = outer_radius*(inner_radius/outer_radius + pow(std::abs(p[0]), power));

    dealii::Point<2> q = p;
    q[0] = -radius*cos(angle);
    q[1] = radius*sin(angle);
    return q;
}

void half_cylinder(dealii::parallel::distributed::Triangulation<2> & tria,
                   const unsigned int n_cells_circle,
                   const unsigned int n_cells_radial)
{
    //const double pi = dealii::numbers::PI;
    //double inner_circumference = inner_radius*pi;
    //double outer_circumference = outer_radius*pi;
    //const double rectangle_height = inner_circumference;
    dealii::Point<2> p1(-1,0.0), p2(-0.0,1.0);

    const bool colorize = true;

    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_cells_radial;
    n_subdivisions[1] = n_cells_circle;
    dealii::GridGenerator::subdivided_hyper_rectangle (tria, n_subdivisions, p1, p2, colorize);

    dealii::GridTools::transform (&warp_cylinder, tria);

    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, dealii::SphericalManifold<2>(center));

    // Assign BC
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell) {
        //if (!cell->is_locally_owned()) continue;
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    cell->face(face)->set_boundary_id (1004); // x_left, Farfield
                } else if (current_id == 1) {
                    cell->face(face)->set_boundary_id (1001); // x_right, Symmetry/Wall
                } else if (current_id == 2) {
                    cell->face(face)->set_boundary_id (1001); // y_bottom, Symmetry/Wall
                } else if (current_id == 3) {
                    cell->face(face)->set_boundary_id (1001); // y_top, Wall
                } else {
                    std::abort();
                }
            }
        }
    }
}

template <int dim, int nstate>
EulerCylinder<dim,nstate>::EulerCylinder(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerCylinder<dim,nstate>
::run_test () const
{
    pcout << " Running Euler cylinder entropy convergence. " << std::endl;
    static_assert(dim==2);
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type == param.PartialDifferentialEquation::euler, dealii::ExcMessage("Can't run Euler case if PDE is not Euler"));
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input;

        std::vector<double> entropy_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        // Generate grid and mapping
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        const unsigned int n_cells_circle = n_1d_cells[0];
        const unsigned int n_cells_radial = 1.5*n_cells_circle;
        half_cylinder(*grid, n_cells_circle, n_cells_radial);

        // Create DG object
        const int solution_degree = poly_degree;
        //const int grid_degree = std::max(2,solution_degree);
        const int grid_degree = solution_degree+1;
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

        dg->allocate_system ();
        // Initialize coarse grid solution with free-stream
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping(poly_degree);

        dealii::Vector<float> estimated_error_per_cell(grid->n_active_cells());
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            // Interpolate solution from previous grid
            if (igrid>0) {
                dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
                dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
                solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
                dg->high_order_grid->prepare_for_coarsening_and_refinement();

                grid->refine_global (1);
                //dealii::GridRefinement::refine_and_coarsen_fixed_number(*grid,
                //                                estimated_error_per_cell,
                //                                0.3,
                //                                0.03);
                //grid->execute_coarsening_and_refinement();
                dg->high_order_grid->execute_coarsening_and_refinement();
                dg->allocate_system ();
                dg->solution.zero_out_ghosts();
                solution_transfer.interpolate(dg->solution);
                dg->solution.update_ghost_values();
            }

            // std::string filename = "grid_cylinder-" + dealii::Utilities::int_to_string(igrid, 1) + ".eps";
            // std::ofstream out (filename);
            // dealii::GridOut grid_out;
            // grid_out.write_eps (*grid, out);
            // pcout << " Grid #" << igrid+1 << " . Number of cells: " << grid->n_global_active_cells() << std::endl;
            // pcout << " written to " << filename << std::endl << std::endl;


            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            //ode_solver->initialize_steady_polynomial_ramping (poly_degree);
            // Solve the steady state problem
            ode_solver->steady_state();

            const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
            dealii::hp::MappingCollection<dim> mapping_collection(mapping);
            dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, dealii::update_values | dealii::update_JxW_values); ///< FEValues of volume.
            // Overintegrate the error to make sure there is not integration error in the error estimate
            //int overintegrate = 0;
            //dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            //dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra,
            //        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            std::array<double,nstate> soln_at_q;

            double l2error = 0;
            double area = 0;
            const double exact_area = (std::pow(outer_radius+inner_radius, 2.0) - std::pow(inner_radius,2.0))*dealii::numbers::PI / 2.0;


            const double entropy_inf = euler_physics_double.entropy_inf;

            estimated_error_per_cell.reinit(grid->n_active_cells());
            // Integrate solution error and output error
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                const int i_fele = cell->active_fe_index();
                const int i_quad = i_fele;
                const int i_mapp = 0;

                fe_values_collection_volume.reinit (cell, i_quad, i_mapp, i_fele);
                const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

                //fe_values_volume.reinit (cell);
                std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_volume.dofs_per_cell);
                cell->get_dof_indices (dofs_indices);

                double cell_l2error = 0;
                for (unsigned int iquad=0; iquad<fe_values_volume.n_quadrature_points; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_volume.shape_value_component(idof, iquad, istate);
                    }
                    const double entropy = euler_physics_double.compute_entropy_measure(soln_at_q);

                    const double uexact = entropy_inf;
                    cell_l2error += pow(entropy - uexact, 2) * fe_values_volume.JxW(iquad);
                    estimated_error_per_cell[cell->active_cell_index()] = cell_l2error;
                    l2error += cell_l2error;

                    area += fe_values_volume.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
            const double area_mpi_sum = dealii::Utilities::MPI::sum(area, mpi_communicator);

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            entropy_error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("L2_entropy_error", l2error_mpi_sum);
            convergence_table.add_value("area_error", std::abs(area_mpi_sum-exact_area));


            pcout << " Grid size h: " << dx
                 << " L2-entropy_error: " << l2error_mpi_sum
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(entropy_error[igrid]/entropy_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  entropy_error1 " << entropy_error[igrid-1]
                     << "  entropy_error2 " << entropy_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
            }

            //output_results (igrid);
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("L2_entropy_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("area_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("L2_entropy_error", true);
        convergence_table.set_scientific("area_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(entropy_error[n_grids-1]/entropy_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        double before_last_slope = last_slope;
        if ( n_grids > 2 ) {
        before_last_slope = log(entropy_error[n_grids-2]/entropy_error[n_grids-3])
                            / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        }
        const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -0.1;
            pcout << std::endl
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

#if PHILIP_DIM==2
    template class EulerCylinder <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


