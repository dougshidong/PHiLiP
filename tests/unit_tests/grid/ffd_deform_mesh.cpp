#include <deal.II/grid/grid_generator.h>

#include "mesh/high_order_grid.h"
#include "mesh/free_form_deformation.h"

/// Tests the Free-Form Deformation function deform_mesh()

/** Tests the mesh movement by moving the mesh and integrating its volume.
 *  Furthermore, it checks that the surface displacements resulting from the mesh movement
 *  are consistent with the prescribed surface displacements.
 */
int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    const int initial_n_cells = 10;
    const unsigned int n_grids = 1;
    const unsigned int p_start = 3;
    const unsigned int p_end = 3;
    const double amplitude = 0.1;
    const double tpi = 2*std::atan(1)*4;

    unsigned int iffd_output = 0;
    const unsigned int ni_ffd_start = 11;
    const unsigned int ni_ffd_end = 101;
    const unsigned int ni_ffd_interval = 10;
    for (unsigned int ni_ffd = ni_ffd_start; ni_ffd <= ni_ffd_end; ni_ffd+=ni_ffd_interval) {

        double last_error = 1e+300;

        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

            for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

                using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                    MPI_COMM_WORLD,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening));

                dealii::GridGenerator::subdivided_hyper_cube(*grid, initial_n_cells);


                HighOrderGrid<dim,double> high_order_grid(poly_degree, grid);

                for (unsigned int i=0; i<igrid; ++i) {
                    high_order_grid.prepare_for_coarsening_and_refinement();
                    grid->refine_global (1);
                    high_order_grid.execute_coarsening_and_refinement();
                }
                const int n_refine = 2;
                for (int i=0; i<n_refine;i++) {
                    high_order_grid.prepare_for_coarsening_and_refinement();
                    grid->prepare_coarsening_and_refinement();
                    unsigned int icell = 0;
                    for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
                        if (!cell->is_locally_owned()) continue;
                        icell++;
                        if (icell > grid->n_active_cells()/2) {
                            cell->set_refine_flag();
                        }
                    }
                    grid->execute_coarsening_and_refinement();
                    bool mesh_out = (i==n_refine-1);
                    high_order_grid.execute_coarsening_and_refinement(mesh_out);
                }

                high_order_grid.prepare_for_coarsening_and_refinement();
                grid->repartition();
                high_order_grid.execute_coarsening_and_refinement();

                high_order_grid.reset_initial_nodes();
                high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

                const dealii::Point<dim> ffd_origin(0.0,0.0);
                const std::array<double,dim> ffd_rectangle_lengths = {{1.0,0.5}};
                const unsigned int nj_ffd = 2;
                const std::array<unsigned int,dim> ffd_ndim_control_pts = {{ni_ffd,nj_ffd}};
                FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

                ffd.output_ffd_vtu(iffd_output++);

                for (unsigned int i_ffd = 0; i_ffd < ffd_ndim_control_pts[0]; ++i_ffd) {
                    unsigned int j_ffd = 0;
                    const std::array<unsigned int,dim> ijk_ffd = {{i_ffd, j_ffd}};
                    const unsigned int ictl_ffd = ffd.grid_to_global(ijk_ffd);
                    const dealii::Point<dim> old_ffd_point = ffd.control_pts[ictl_ffd];
                    dealii::Point<dim> new_ffd_point = old_ffd_point;
                    new_ffd_point[1] = amplitude*std::sin(old_ffd_point[0]*tpi);
                    ffd.move_ctl_dx ( ijk_ffd, new_ffd_point - old_ffd_point);
                }
                ffd.output_ffd_vtu(iffd_output++);
                ffd.deform_mesh(high_order_grid);
                high_order_grid.update_surface_nodes();
                high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

                auto surf_node = high_order_grid.locally_relevant_surface_points.begin();
                auto init_surf_node = high_order_grid.initial_locally_relevant_surface_points.begin();
                double local_err = 0.0;
                for (; surf_node != high_order_grid.locally_relevant_surface_points.end(); ++surf_node, ++init_surf_node) {
                    if ((*init_surf_node)[1] == 0.0) {
                        double x = (*surf_node)[0];
                        double y = (*surf_node)[1];
                        double real_y = amplitude*std::sin(x*tpi);
                        double diff = y - real_y;
                        local_err += diff*diff;
                    }
                }
                const double error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(local_err, MPI_COMM_WORLD));

                const unsigned int nx_surf = high_order_grid.surface_nodes.size() / 4;
                const double error = error_mpi_sum / nx_surf;

                pcout << " error: " << error << std::endl;

                if (error > last_error) fail_bool = true;
                last_error = error;

            }
        }
    }

    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}

