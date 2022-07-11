#ifndef __HALF_CYLINDER_H__
#define __HALF_CYLINDER_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

void half_cylinder(dealii::parallel::distributed::Triangulation<2> & tria,
                   const unsigned int n_cells_circle,
                   const unsigned int n_cells_radial);

} // namespace Grids
} // namespace PHiLiP
#endif

