#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "non_periodic_cube.h"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void non_periodic_cube(
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
            std::cout << "cell" << std::endl;
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (shu_osher) {
                // Set left boundary to Riemann Far Field condition
                if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(1004);
                if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
            } else if (shock_tube) {
                // Set left boundary to Wall Boundary
                if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(1001);
                if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
            }
        }
    }
}

#if PHILIP_DIM==1
template void non_periodic_cube<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher);
#else
template void non_periodic_cube<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher);
#endif
} // namespace Grids
} // namespace PHiLiP
