#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <stdlib.h>
#include <iostream>

#include "flat_plate_cube.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void flat_plate_cube(std::shared_ptr<TriangulationType> &grid,
                     const double /*domain_left*/,
                     const double /*domain_right*/,
                     const int number_of_cells_per_direction)
{
    if constexpr(dim==2) {
        const int number_of_refinements = log(number_of_cells_per_direction)/log(2);
    
        dealii::Point<2,double> left_corner,right_corner;
        left_corner[0] = -0.5;
        left_corner[1] = 0.0;
        right_corner[0] = 1.0;
        right_corner[1] = 1.0;
        const bool colorize = true;
        dealii::GridGenerator::hyper_rectangle(*grid, left_corner, right_corner, colorize);
        grid->refine_global(number_of_refinements);
        for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 0) cell->face(face)->set_boundary_id (1003);
                    if (current_id == 1) cell->face(face)->set_boundary_id (1002);
                    if (current_id == 2) {
                        if (cell->face(face)->center()[0]<=0.0){
                            cell->face(face)->set_boundary_id (1005);
                        }else{
                            cell->face(face)->set_boundary_id (1006);
                        }
                    }
                    if (current_id == 3) cell->face(face)->set_boundary_id (1005);
                }
            }
            
        }
    }
}

#if PHILIP_DIM==1
    template void flat_plate_cube<PHILIP_DIM, dealii::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif
#if PHILIP_DIM!=1
    template void flat_plate_cube<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif

} // namespace Grids
} // namespace PHiLiP