#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "mesh/grids/naca_airfoil_grid.hpp"
#include "euler_naca0012.hpp"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerNACA0012<dim,nstate>::EulerNACA0012(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerNACA0012<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim == 2, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
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
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

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

        std::vector<unsigned int> n_subdivisions(dim);
        n_subdivisions[1] = n_1d_cells[0]; // y-direction
        n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

        // const double channel_length = 3.0;
        // const double channel_height = 0.8;
        // Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);

        // const double solution_degree = poly_degree;
        // const double grid_degree = 3;
        // // Create DG object
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, &grid);

        // // Initialize coarse grid solution with free-stream
        // dg->allocate_system ();
        // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // // Create ODE solver and ramp up the solution from p0
        // std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        // ode_solver->initialize_steady_polynomial_ramping (poly_degree);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            //if (igrid!=0) {
            //    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
            //    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
            //    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
            //    dg->high_order_grid.prepare_for_coarsening_and_refinement();
            //    grid->refine_global (1);
            //    dg->high_order_grid.execute_coarsening_and_refinement(true);
            //    dg->allocate_system ();
            //    dg->solution.zero_out_ghosts();
            //    solution_transfer.interpolate(dg->solution);
            //    dg->solution.update_ghost_values();
            //}

            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

            dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
            airfoil_data.airfoil_type = "NACA";
            airfoil_data.naca_id      = "0012";
            airfoil_data.airfoil_length = 1.0;
            airfoil_data.height         = 150.0; // Farfield radius.
            airfoil_data.length_b2      = 150.0;
            airfoil_data.incline_factor = 0.0;
            airfoil_data.bias_factor    = 5.0;
            airfoil_data.refinements    = 0;
            airfoil_data.n_subdivision_x_0 = 15;
            airfoil_data.n_subdivision_x_1 = 7;
            airfoil_data.n_subdivision_x_2 = 15;
            airfoil_data.n_subdivision_y = 10;
            airfoil_data.airfoil_sampling_factor = 3; // default 2

            // dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
            // // Assign a manifold to have curved geometry
            // unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
            // grid->reset_all_manifolds();
            // grid->set_all_manifold_ids(manifold_id);
            // // // Set Flat manifold on the domain, but not on the boundary.
            // grid->set_manifold(manifold_id, dealii::FlatManifold<2>());

            // // Set boundary type and design type
            // for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
            //     for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            //         if (cell->face(face)->at_boundary()) {
            //             unsigned int current_id = cell->face(face)->boundary_id();
            //             if (current_id == 0 || current_id == 1) {
            //                 cell->face(face)->set_boundary_id (1004); // farfield
            //             } else {
            //                 cell->face(face)->set_boundary_id (1001); // Outflow with supersonic or back_pressure
            //             }
            //         }
            //     }
            // }

            n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
            n_subdivisions[1] = airfoil_data.n_subdivision_y;
            //Grids::naca_airfoil(*grid, airfoil_data.naca_id, n_subdivisions, airfoil_data.height);
            Grids::naca_airfoil(*grid, airfoil_data);

            const double solution_degree = poly_degree;
            const double grid_degree = solution_degree+1;
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

            dg->high_order_grid.prepare_for_coarsening_and_refinement();
            grid->refine_global (1);
            dg->high_order_grid.execute_coarsening_and_refinement(true);

            // Initialize coarse grid solution with free-stream
            dg->allocate_system ();
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Create ODE solver and ramp up the solution from p0
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
            ode_solver->initialize_steady_polynomial_ramping (poly_degree);

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            //dealii::MappingQ<dim> mapping(dg->max_degree+overintegrate);
            //const dealii::MappingManifold<dim,dim> mapping;
            const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid.mapping_fe_field));
            dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra,
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0;


            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

            const double entropy_inf = euler_physics_double.entropy_inf;

            // Integrate solution error and output error
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }
                    const double entropy = euler_physics_double.compute_entropy_measure(soln_at_q);

                    const double uexact = entropy_inf;
                    l2error += pow(entropy - uexact, 2) * fe_values_extra.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));


            // Convergence table
            double dx = 1.0/pow(n_dofs,(1.0/dim));
            //dx = dealii::GridTools::maximal_cell_diameter(*grid);
            grid_size[igrid] = dx;
            entropy_error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("L2_entropy_error", l2error_mpi_sum);


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
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("L2_entropy_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(entropy_error[n_grids-1]/entropy_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        //double before_last_slope = last_slope;
        //if ( n_grids > 2 ) {
        //    before_last_slope = log(entropy_error[n_grids-2]/entropy_error[n_grids-3])
        //                        / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        //}
        //const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_avg = last_slope;
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
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


