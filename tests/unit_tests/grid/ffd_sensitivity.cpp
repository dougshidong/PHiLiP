#include <deal.II/grid/grid_generator.h>

#include "mesh/high_order_grid.h"
#include "mesh/free_form_deformation.h"

/// Tests the Free-Form Deformation function deform_mesh()

/** Tests the mesh movement by moving the mesh and integrating its volume.
 *  Furthermore, it checks that the surface displacements resulting from the mesh movement
 *  are consistent with the prescribed surface displacements.
 */

const double EPS = 1e-4;
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

                    //const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
                    for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

                        // if (   ijk[0] == 0 // Constrain first column of FFD points.
                        //     || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                        //     || d_ffd == 0 // Constrain x-direction of FFD points.
                        //    ) {
                        //     continue;
                        // }
                        ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
                    }
                }
                const unsigned int n_design_variables = ffd_design_variables_indices_dim.size();

                // Get analytical dXvdXp
                dealii::TrilinosWrappers::SparseMatrix dXvdXp, dXvdXp_FD;
                ffd.get_dXvdXp(high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
                ffd.get_dXvdXp_FD(high_order_grid, ffd_design_variables_indices_dim, dXvdXp_FD, EPS);

                const double dXvdXp_frob_norm = dXvdXp.frobenius_norm();

                dXvdXp.add(-1.0, dXvdXp_FD);

                const double abs_diff_frob_norm = dXvdXp.frobenius_norm();
                const double rel_diff_frob_norm = abs_diff_frob_norm / dXvdXp_frob_norm;

                pcout << " ****************************** " << std::endl;
                pcout << " dXvdXp error: " << rel_diff_frob_norm << std::endl;
                pcout << " ****************************** " << std::endl;

                if (rel_diff_frob_norm > 1e-4) fail_bool = true;


                std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvsdXp_vector_AD = ffd.get_dXvsdXp ( high_order_grid, ffd_design_variables_indices_dim );
                std::vector<dealii::LinearAlgebra::distributed::Vector<double>> dXvsdXp_vector_FD = ffd.get_dXvsdXp_FD ( high_order_grid, ffd_design_variables_indices_dim, EPS );
                dealii::TrilinosWrappers::SparseMatrix dXvsdXp_matrix;
                ffd.get_dXvsdXp ( high_order_grid, ffd_design_variables_indices_dim, dXvsdXp_matrix );

                double local_matrix_error = 0.0;
                const auto &row_part = dXvsdXp_matrix.locally_owned_range_indices();
                for (unsigned int i_design = 0; i_design < n_design_variables; ++i_design) {
                    auto diff = dXvsdXp_vector_AD[i_design];
                    diff -= dXvsdXp_vector_FD[i_design];

                    const double dXvsdXp_AD_l2_norm = dXvsdXp_vector_AD[i_design].l2_norm(); (void) dXvsdXp_AD_l2_norm;
                    const double dXvsdXp_FD_l2_norm = dXvsdXp_vector_FD[i_design].l2_norm(); (void) dXvsdXp_FD_l2_norm;
                    const double abs_diff_norm = diff.l2_norm();
                    const double rel_diff_norm = (dXvsdXp_AD_l2_norm == 0.0) ? abs_diff_norm : abs_diff_norm / dXvsdXp_AD_l2_norm;
                    pcout << " dXvsdXp i: " << i_design << " error: " << rel_diff_norm << std::endl;

                    if (rel_diff_norm > 1e-4) fail_bool = true;


                    for (const auto &row : row_part) {
                        const double val_m = dXvsdXp_matrix.el(row,i_design);
                        const double val_v = dXvsdXp_vector_AD[i_design][row];
                        const double diff_mv = val_m - val_v;

                        local_matrix_error += diff_mv*diff_mv;
                    }
                }
                double matrix_abs_error = std::sqrt(dealii::Utilities::MPI::sum(local_matrix_error,MPI_COMM_WORLD));
                double matrix_rel_error = matrix_abs_error / dXvsdXp_matrix.frobenius_norm();
                pcout << " dXvsdXp_vector - dXvsdXp_matrix: " << matrix_rel_error << std::endl;
                if (matrix_rel_error > 1e-12) fail_bool = true;
            }
        }
    }

    if (fail_bool) {
        pcout << "Test failed. Error from analytical sensitivity does not match finite difference." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}

