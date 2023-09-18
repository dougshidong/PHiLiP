#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "non_periodic_cube.h"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void non_periodic_cube_flow(
    TriangulationType& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher) 
{
    dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);

    if (shock_tube || shu_osher) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy boundary ID
            cell->set_material_id(9002);
            if (cell == grid.begin_active())
            {
                for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (shu_osher) {
                        // Set left boundary to Riemann Far Field condition
                        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(1004);
                    }
                    else {
                        // Set left boundary to Wall Boundary
                        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(1001);
                    }
                }
            }
            else if (cell == grid.end())
            {
                for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    // Set left boundary to Wall Boundary
                    if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(1001);
                }
            }
        }
    }
}

template void non_periodic_cube_flow<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher);

template void non_periodic_cube_flow<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher);

} // namespace Grids
} // namespace PHiLiP
