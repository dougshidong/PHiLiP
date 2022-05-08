#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <exception>

#include <deal.II/fe/mapping_fe_field.h> 

#include "mesh/high_order_grid.h"

using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;

dealii::Point<2> center(0.0,0.0);
const double inner_radius = 1, outer_radius = inner_radius*20;

dealii::Point<2> warp_cylinder (const dealii::Point<2> &p)//, const double inner_radius, const double outer_radius)
{
    const double rectangle_height = 1.0;
    //const double original_radius = std::abs(p[0]);
    const double angle = p[1]/rectangle_height * dealii::numbers::PI;

    //const double radius = std::abs(p[0]);

    const double power = 2.25;
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
    dealii::Point<2> p1(-1,0.0), p2(-0.0,1.0);

    const bool colorize = true;

    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_cells_radial;
    n_subdivisions[1] = n_cells_circle;
    dealii::GridGenerator::subdivided_hyper_rectangle (tria, n_subdivisions, p1, p2, colorize);

    dealii::GridTools::transform (&warp_cylinder, tria);

    tria.set_all_manifold_ids(1);
    tria.set_all_manifold_ids_on_boundary(0);
    tria.set_all_manifold_ids(0);
    dealii::TransfiniteInterpolationManifold<2> inner_manifold;
    inner_manifold.initialize(tria);
    tria.set_manifold(0, dealii::SphericalManifold<2>(center));
    tria.set_manifold(1, inner_manifold);


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

int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    const unsigned int p_start = 1;
    const unsigned int p_end   = 4;

    const unsigned int n_grids = 3;

    bool has_invalid_poly = false;
    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // Generate grid and mapping
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        // This grid is known to generated a negative Jacobian as seen
        // in the EulerCylinder test case described in 
        // https://github.com/dougshidong/PHiLiP/issues/27
        const unsigned int n_cells_circle = 2;
        const unsigned int n_cells_radial = 2*n_cells_circle;
        half_cylinder(*grid, n_cells_circle, n_cells_radial);

        PHiLiP::HighOrderGrid<dim,double> high_order_grid(poly_degree, grid);

        //bool has_invalid_grid = false;
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            // Interpolate solution from previous grid
            if (igrid>0) {
                high_order_grid.prepare_for_coarsening_and_refinement();
                grid->refine_global (1);
                high_order_grid.execute_coarsening_and_refinement();

                high_order_grid.prepare_for_coarsening_and_refinement();
                grid->repartition();
                high_order_grid.execute_coarsening_and_refinement(true);
            }

            auto cell = high_order_grid.dof_handler_grid.begin_active();
            auto endcell = high_order_grid.dof_handler_grid.end();

            for (; cell!=endcell; ++cell) {
                if (!cell->is_locally_owned())  continue;
                const bool is_invalid_cell = high_order_grid.check_valid_cell(cell);

                if ( !is_invalid_cell ) {
                    std::cout << " Poly: " << poly_degree
                              << " Grid: " << igrid
                              << " Cell: " << cell->active_cell_index() << " has an invalid Jacobian." << std::endl;
                    //has_invalid_grid = true;
     bool fixed_invalid_cell = high_order_grid.fix_invalid_cell(cell);
     if (fixed_invalid_cell) std::cout << "Fixed it." << std::endl;
                    else has_invalid_poly = true;
                }
            }
            high_order_grid.volume_nodes.update_ghost_values();
            if (has_invalid_poly) std::abort();
        }
    }

    bool mpi_has_invalid_poly;
    MPI_Allreduce(&has_invalid_poly, &mpi_has_invalid_poly, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

    return mpi_has_invalid_poly;
}

