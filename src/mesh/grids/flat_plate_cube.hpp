#ifndef __FLAT_PLATE_CUBE_H__
#define __FLAT_PLATE_CUBE_H__

#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a straight edge flat plate cube mesh
template<int dim, typename TriangulationType>
void flat_plate_cube(std::shared_ptr<TriangulationType> &grid,
                     std::shared_ptr<TriangulationType> &sub_grid_1,
                     std::shared_ptr<TriangulationType> &sub_grid_2,
                     const double free_length,
                     const double free_height,
                     const double plate_length,
                     const double skewness_x_free,
                     const double skewness_x_plate,
                     const double skewness_y,
                     const int number_of_subdivisions_in_x_direction_free,
                     const int number_of_subdivisions_in_x_direction_plate,
                     const int number_of_subdivisions_in_y_direction);

} // namespace Grids
} // namespace PHiLiP
#endif