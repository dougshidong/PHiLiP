#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "half_cylinder.hpp"

namespace PHiLiP {
namespace Grids {


namespace {

dealii::Point<2> center(0.0,0.0);
const double inner_radius = 0.5;
const double outer_radius = inner_radius*40;

dealii::Point<2> warp_cylinder (const dealii::Point<2> &p)//, const double inner_radius, const double outer_radius)
{
    const double rectangle_height = 1.0;
    //const double original_radius = std::abs(p[0]);
    const double angle = p[1]/rectangle_height * dealii::numbers::PI;

    //const double radius = std::abs(p[0]);

    // Radial grid progression.
    const double power = 1.8;
    const double radius = outer_radius*(inner_radius/outer_radius + pow(std::abs(p[0]), power));

    dealii::Point<2> q = p;
    q[0] = -radius*cos(angle);
    q[1] = radius*sin(angle);
    return q;
}

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

} // namespace Grids
} // namespace PHiLiP
