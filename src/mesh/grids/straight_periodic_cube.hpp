#ifndef __STRAIGHT_PERIODIC_CUBE_H__
#define __STRAIGHT_PERIODIC_CUBE_H__

#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a straight edge triply periodic cube mesh
template<int dim, typename TriangulationType>
void straight_periodic_cube(std::shared_ptr<TriangulationType> &grid,
                            const double domain_left,
                            const double domain_right,
                            const int number_of_cells_per_direction);

} // namespace Grids
} // namespace PHiLiP
#endif

