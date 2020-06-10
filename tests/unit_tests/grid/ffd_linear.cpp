#include <deal.II/grid/grid_generator.h>

#include "mesh/high_order_grid.h"
#include "mesh/free_form_deformation.h"

/// Tests the Free-Form Deformation sensitivities to make sure it is indeed linear.
/** 1. Evaluate dXvsdXp_1 for a set of control points.
 *  2. Displace control points
 *  3. Evaluate dXvsdXp_2 for displaced set.
 *  4. Compare dXvsdXp_1 to dXvsdXp_2.
 */

const double TOL = 1e-15;
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
    const unsigned int p_start = 1;
    const unsigned int p_end = 1;

    unsigned int iffd_output = 0;
    const unsigned int ni_ffd_start = 11;
    const unsigned int ni_ffd_end = 11;
    const unsigned int ni_ffd_interval = 10;

    const dealii::Point<dim> ffd_origin(-0.01,-0.01);
    const std::array<double,dim> ffd_rectangle_lengths = {1.0,0.5};
    const unsigned int nj_ffd = 3;

    for (unsigned int ni_ffd = ni_ffd_start; ni_ffd <= ni_ffd_end; ni_ffd+=ni_ffd_interval) {

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

                const std::array<unsigned int,dim> ffd_ndim_control_pts = {ni_ffd,nj_ffd};
                FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

                ffd.output_ffd_vtu(iffd_output++);

                // List of FFD design points (all of them)
                std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
                for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {
                    for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

                        ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
                    }
                }

                auto old_control_pts = ffd.control_pts;
                dealii::TrilinosWrappers::SparseMatrix dXvsdXp_1, dXvsdXp_2;
                ffd.get_dXvsdXp(high_order_grid, ffd_design_variables_indices_dim, dXvsdXp_1);

                for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {
                        dealii::Tensor<1,dim,double> random_displacement;
                        for (int d=0;d<dim;++d) {
                            const double fMin = 0.0, fMax = 0.01;
                            double f = (double)rand() / RAND_MAX;
                            double displacement = fMin + f * (fMax - fMin);
                            random_displacement[d] = displacement;
                        }
                        ffd.move_ctl_dx ( i_ctl, random_displacement );
                }

                ffd.get_dXvsdXp(high_order_grid, ffd_design_variables_indices_dim, dXvsdXp_2);

                const double dXvsdXp_frob_norm = dXvsdXp_1.frobenius_norm();

                dXvsdXp_1.add(-1.0, dXvsdXp_2);

                const double abs_diff_frob_norm = dXvsdXp_1.frobenius_norm();
                const double rel_diff_frob_norm = abs_diff_frob_norm / dXvsdXp_frob_norm;

                double diff = 0.0;
                for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {
                    for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {
                        diff = (old_control_pts[i_ctl] - ffd.control_pts[i_ctl]).norm();
                    }
                }
                pcout << " ****************************** " << std::endl;
                pcout << " Diff in control vector (non-zero): " << diff << std::endl;
                pcout << " ****************************** " << std::endl;
                pcout << " ****************************** " << std::endl;
                pcout << " dXvsdXp error: " << rel_diff_frob_norm << std::endl;
                pcout << " ****************************** " << std::endl;

                if (rel_diff_frob_norm > TOL) fail_bool = true;

            }
        }
    }

    if (fail_bool) {
        pcout << "Test failed. FFD sensitivity is not linear." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}

