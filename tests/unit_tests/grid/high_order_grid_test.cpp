#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <exception>
#include <deal.II/fe/mapping.h> 
#include <deal.II/base/exceptions.h> // ExcTransformationFailed

#include <deal.II/fe/mapping_fe_field.h> 
#include <deal.II/fe/mapping_q.h> 

#include "mesh/high_order_grid.h"

// https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Recursions
template <int dim>
double volume_n_ball(const double radius)
{
    const double pi = dealii::numbers::PI;
    return (2.0*pi*radius*radius/dim)*volume_n_ball<dim-2>(radius);
}
template <> double volume_n_ball<0>(const double /*radius*/) { return 1.0; }
template <> double volume_n_ball<1>(const double radius) { return 2.0*radius; }

int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    const unsigned int p_start = 1; // Must be at least order 1
    const unsigned int p_end   = 4;
    const unsigned int n_grids = 3;
    //const std::vector<int> n_1d_cells = {2,4,8,16};

    //const unsigned int n_cells_circle = n_1d_cells[0];
    //const unsigned int n_cells_radial = 3*n_cells_circle;

    dealii::Point<dim> center; // Constructor initializes Point at the origin
    const double inner_radius = 1, outer_radius = inner_radius*10;
    const double exact_volume = (volume_n_ball<dim>(outer_radius) - volume_n_ball<dim>(inner_radius)) / std::pow(2,dim);

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // Generate the original grid and assign a manifold to it
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        const int n_cells = 0;
        dealii::GridGenerator::quarter_hyper_shell<dim>(*grid, center, inner_radius, outer_radius, n_cells);//, n_cells = 0, colorize = false);
        // Set a spherical manifold
        // Works but we will usually not have a volume manifold that we can provide.
        //grid->set_all_manifold_ids(0);
        //grid->set_manifold(0, dealii::SphericalManifold<dim>(center));

        // Set a spherical manifold on the boundary and a TransfiniteInterpolationManifold in the domain
        // This is more realistic with what we will be doing since we will usually provide and boundary parametrization
        // but have no idea of the volume parametrization. The initial curving of the TransfiniteInterpolationManifold
        // will ensure that no cells initially have a negative Jacobians. This curvature will be transfered to a polynomial
        // representation using MappingFEField
        grid->set_all_manifold_ids(1);
        grid->set_all_manifold_ids_on_boundary(0);
        grid->set_manifold(0, dealii::SphericalManifold<dim>(center));
        dealii::TransfiniteInterpolationManifold<dim> transfinite_interpolation;
        transfinite_interpolation.initialize(*grid);
        grid->set_manifold(1, transfinite_interpolation);


        HighOrderGrid<dim,double> high_order_grid(poly_degree, grid);
        //std::shared_ptr < dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> > mapping = (high_order_grid.mapping_fe_field);
        auto mapping = (high_order_grid.mapping_fe_field);
        grid->reset_all_manifolds();

        dealii::ConvergenceTable convergence_table;
        std::vector<double> grid_size(n_grids);
        std::vector<double> volume_error(n_grids);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            high_order_grid.prepare_for_coarsening_and_refinement();

            grid->refine_global (1);
            //int icell = 0;
            //for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
            //    if (!cell->is_locally_owned()) continue;
            //    icell++;
            //    if (icell < 3) {
            //        cell->set_refine_flag();
            //    } else if (icell%5 == 0) {
            //        cell->set_refine_flag();
            //    } else if (icell%3 == 0) {
            //        cell->set_coarsen_flag();
            //    }
            //}
            //grid->execute_coarsening_and_refinement();

            high_order_grid.execute_coarsening_and_refinement();

            high_order_grid.prepare_for_coarsening_and_refinement();
            grid->repartition();
            high_order_grid.execute_coarsening_and_refinement(true);

            const unsigned int n_dofs = high_order_grid.dof_handler_grid.n_dofs();

            const unsigned int n_global_active_cells = grid->n_global_active_cells();

            // Output for Paraview visualization
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler(high_order_grid.dof_handler_grid);
            std::vector<std::string> solution_names;
            for (int d=0;d<dim;++d) {
                if (d==0) solution_names.push_back("x");
                if (d==1) solution_names.push_back("y");
                if (d==2) solution_names.push_back("z");
            }
            data_out.add_data_vector(high_order_grid.volume_nodes, solution_names);

            const int iproc = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
            data_out.build_patches(*mapping, poly_degree, dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
            std::string filename = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
            filename += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
            filename += dealii::Utilities::int_to_string(igrid, 4) + ".";
            filename += dealii::Utilities::int_to_string(iproc, 4);
            filename += ".vtu";
            std::ofstream output(filename);
            data_out.write_vtu(output);

            if (iproc == 0) {
                std::vector<std::string> filenames;
                for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++iproc) {
                    std::string fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
                    fn += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
                    fn += dealii::Utilities::int_to_string(igrid, 4) + ".";
                    fn += dealii::Utilities::int_to_string(iproc, 4);
                    fn += ".vtu";
                    filenames.push_back(fn);
                }
                std::string master_fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
                master_fn += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
                master_fn += dealii::Utilities::int_to_string(igrid, 4) + ".pvtu";
                std::ofstream master_output(master_fn);
                data_out.write_pvtu_record(master_output, filenames);
            }


            // Perform a mesh deformation
            // This basically translates all the volume_nodes by 1.0 in every direction
            // This translation should not affect the volume estimate
            for (auto dof = high_order_grid.volume_nodes.begin(); dof != high_order_grid.volume_nodes.end(); ++dof) {
                *dof += 1.0;
            }
            high_order_grid.volume_nodes.update_ghost_values();
            
            // This grid transformation is not necessary, it is simply to prove a point that once we use MappingFEField,
            // the Triangulation's vertices locations become irrelevant. All that matters is the cell to cell connectivity.
            // This is an extreme example where we basically set all the vertices to a single point x = y = z = 1.0.
            // See discussion in the following Github issue:
            // https://github.com/dealii/dealii/issues/8877#issuecomment-536831446
            
            // Weirdly enough, it does affect the p1 geometry discretization. Not sure why.
            // It might be using some of the Triangulation's vertices instead of the MappingFEField volume_nodes?
            dealii::GridTools::transform(
                [](const dealii::Point<dim> &old_point) -> dealii::Point<dim> {
                    dealii::Point<dim> new_point;
                    for (int d=0;d<dim;++d) {
                        new_point[d] = 0.0 * old_point[d] + 1.0;
                    }
                    return new_point;
                },
                *grid);

            // Integrate solution error and output error
            // Overintegrate the error to make sure there is no integration error in the error estimate
            const int overintegrate = 3;
            const unsigned int n_quad_pts_1D = poly_degree+1+overintegrate;
            dealii::QGauss<dim> quadrature(n_quad_pts_1D);
            const unsigned int n_quad_pts = quadrature.size();
            dealii::FEValues<dim,dim> fe_values(*mapping, high_order_grid.get_current_fe_system(), quadrature,
                dealii::update_jacobians
                | dealii::update_JxW_values
                | dealii::update_quadrature_points);
            double total_estimated_volume = 0;
            for (auto cell = high_order_grid.dof_handler_grid.begin_active(); cell!=high_order_grid.dof_handler_grid.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values.reinit (cell);

                double cell_volume = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    cell_volume += fe_values.JxW(iquad);
                }
                total_estimated_volume += cell_volume;
            }
            const double volume_mpi_sum = dealii::Utilities::MPI::sum(total_estimated_volume, MPI_COMM_WORLD);

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            volume_error[igrid] = std::abs(1.0 - volume_mpi_sum/exact_volume);

            pcout << "P = " << poly_degree << " NCells = " << n_global_active_cells
                << " Estimated volume: " << volume_mpi_sum << " Exact Volume: " << exact_volume << std::endl;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("volume_error", volume_error[igrid]);
        }
        // Basically, the volume estimate should not change between the refinement and the mesh translation
        const double tolerance = 1e-8;//-std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        for (unsigned int igrid = 1; igrid < n_grids; igrid++) {
            if (poly_degree <= 1) continue; // Does not currently work with p1 grid
            const double error_diff = std::abs(volume_error[igrid-1] - volume_error[igrid]);
            if (error_diff > tolerance) fail_bool = true;
        }

        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        //convergence_table.evaluate_convergence_rates("volume_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("volume_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);
    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}
