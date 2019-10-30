#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

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

#include "dg/high_order_grid.h"
#include "parameters/all_parameters.h"

int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    const unsigned int n_grids = 3;
    const unsigned int p_start = 1;
    const unsigned int p_end = 4;
    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        std::vector<double> area_error(n_grids);
        std::vector<double> grid_size(n_grids);

        dealii::ConvergenceTable convergence_table;
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
            dealii::Triangulation<dim> grid(
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
            dealii::parallel::distributed::Triangulation<dim> grid(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
            const int n_cells = 2;
            dealii::GridGenerator::subdivided_hyper_cube(grid, n_cells);


            HighOrderGrid<dim,double> high_order_grid(&all_parameters, poly_degree, &grid);

            for (unsigned int i=0; i<igrid; ++i) {
                high_order_grid.prepare_for_coarsening_and_refinement();
                grid.refine_global (1);
                high_order_grid.execute_coarsening_and_refinement();
            }
            //const unsigned int n_cell_grid = grid.n_active_cells();
            //for (auto &cell: grid.active_cell_iterators()) {
            //    if (cell->active_cell_index()<n_cell_grid/2) cell->set_refine_flag();
            //}
            //high_order_grid.prepare_for_coarsening_and_refinement();
            //grid.execute_coarsening_and_refinement();
            //high_order_grid.execute_coarsening_and_refinement();

            high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

#if PHILIP_DIM!=1
            high_order_grid.prepare_for_coarsening_and_refinement();
            grid.repartition();
            high_order_grid.execute_coarsening_and_refinement();
#endif

            std::vector<dealii::Tensor<1,dim,double>> point_displacements(high_order_grid.locally_relevant_surface_points.size());
            auto disp = point_displacements.begin();
            auto point = high_order_grid.locally_relevant_surface_points.begin();
            auto point_end = high_order_grid.locally_relevant_surface_points.end();
            for (;point != point_end; ++point, ++disp) {
                const double amplitude = 0.1;
                (*disp) = 0.0;
                (*disp)[0] = amplitude;
                (*disp)[0] *= (*point)[0];
                if(dim>=2) {
                    (*disp)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[1]);
                }
                if(dim>=3) {
                    (*disp)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[2]);
                }
                //(*disp)[0] *= 1*(*point)[0];
            }

            std::vector<dealii::types::global_dof_index> surface_node_global_indices(dim*high_order_grid.locally_relevant_surface_points.size());
            std::vector<double> surface_node_displacements(dim*high_order_grid.locally_relevant_surface_points.size());
            {
                int inode = 0;
                for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
                    for (unsigned int d=0;d<dim;++d) {
                        const std::pair<unsigned int, unsigned int> point_axis = std::make_pair(ipoint,d);
                        const dealii::types::global_dof_index global_index = high_order_grid.point_and_axis_to_global_index[point_axis];
                        surface_node_global_indices[inode] = global_index;
                        surface_node_displacements[inode] = point_displacements[ipoint][d];
                        //std::cout << surface_node_global_indices[inode] << " " << surface_node_displacements[inode] << std::endl;
                        std::vector<dealii::types::global_dof_index>::iterator it 
                            = std::find(high_order_grid.locally_relevant_surface_nodes_indices.begin(),
                                        high_order_grid.locally_relevant_surface_nodes_indices.end(),
                                        global_index);
                        int index = std::distance(high_order_grid.locally_relevant_surface_nodes_indices.begin(), it);
                        std::cout << surface_node_global_indices[inode] << " " << high_order_grid.locally_relevant_surface_points[ipoint][d] << " ?= " << high_order_grid.local_surface_nodes[index] << std::endl;
                        inode++;
                    }
                }
            }

            std::cout << "Points old" << std::endl;
            for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
                std::cout << high_order_grid.locally_relevant_surface_points[ipoint] << std::endl;
            }
            std::cout << "Points new" << std::endl;
            for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
                std::cout << high_order_grid.locally_relevant_surface_points[ipoint]+point_displacements[ipoint] << std::endl;
            }

#if PHILIP_DIM==1
            using VectorType = dealii::Vector<double>;
#else
            using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
#endif
            MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                meshmover(high_order_grid, surface_node_global_indices, surface_node_displacements);
            VectorType volume_displacements = meshmover.get_volume_displacements();
            high_order_grid.nodes += volume_displacements;

            high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

            const int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(high_order_grid.max_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(high_order_grid.mapping_fe_field), high_order_grid.fe_system, quad_extra, dealii::update_JxW_values);
            double area = 0;
            for (auto cell : high_order_grid.dof_handler_grid.active_cell_iterators()) {
                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                for (unsigned int iquad=0; iquad<quad_extra.size(); ++iquad) {
                    area += fe_values_extra.JxW(iquad);
                }
            }
            const double exact_area = 1.0;
            const double area_mpi_sum = dealii::Utilities::MPI::sum(area, MPI_COMM_WORLD);

            const double dx = 1.0/pow(high_order_grid.dof_handler_grid.n_dofs(),(1.0/dim));
            grid_size[igrid] = dx;
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", grid.n_active_cells());
            convergence_table.add_value("DoFs", high_order_grid.dof_handler_grid.n_dofs());
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("area_error", std::abs(area_mpi_sum-exact_area));

            const double expected_slope = poly_degree+1;
            const double last_slope = log(area_error[n_grids-1]/area_error[n_grids-2])
                                      / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
            double before_last_slope = last_slope;
            if ( n_grids > 2 ) {
                before_last_slope = log(area_error[n_grids-2]/area_error[n_grids-3])
                                    / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
            }
            const double slope_avg = 0.5*(before_last_slope+last_slope);
            const double slope_diff = slope_avg-expected_slope;

            double slope_deficit_tolerance = -0.1;

            if (slope_diff < slope_deficit_tolerance) {
                pcout << std::endl
                     << "Convergence order not achieved. Average last 2 slopes of "
                     << slope_avg << " instead of expected "
                     << expected_slope << " within a tolerance of "
                     << slope_deficit_tolerance
                     << std::endl;
                fail_conv_poly.push_back(poly_degree);
                fail_conv_slop.push_back(slope_avg);
            }

        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("area_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("area_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

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


    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}

