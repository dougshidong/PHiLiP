#ifndef __NON_PERIODIC_CUBE_FLOW_H__
#define __NON_PERIODIC_CUBE_FLOW_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void non_periodic_cube_flow(
    TriangulationType& grid,
    double domain_left,
    double domain_right,
    bool colorize,
    bool shock_tube,
    bool shu_osher);
} // namespace Grids
} // namespace PHiLiP
#endif
