#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "non_periodic_cube.h"

namespace PHiLiP::Grids {

template<int dim, typename TriangulationType>
void non_periodic_cube(
    TriangulationType&  grid,
    double              domain_left,
    double              domain_right,
    bool                colorize,
    const int           left_boundary_id) 
{
    dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);
    std::cout << left_boundary_id << std::endl;
    // Other tests cases set boundary outside of the flow_solver case so 
    // if left_boundary_id is not set, it is skipped so it can be set elsewhere.
    if (left_boundary_id != 9999) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID - Fails without this for some reason
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }
}

#if PHILIP_DIM==1
template void non_periodic_cube<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    double                      domain_left,
    double                      domain_right,
    bool                        colorize,
    const int                   left_boundary_id);
#else
template void non_periodic_cube<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    double                                              domain_left,
    double                                              domain_right,
    bool                                                colorize,
    const int                                           left_boundary_id);
#endif
} // namespace PHiLiP::Grids
