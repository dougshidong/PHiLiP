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
                     const int number_of_subdivisions_in_y_direction)
{
    if constexpr(dim==2) {
        dealii::Point<2,double> 
            sub_grid_1_left_corner,
            sub_grid_1_right_corner,
            sub_grid_2_left_corner,
            sub_grid_2_right_corner;
        sub_grid_1_left_corner[0] = -free_length;
        sub_grid_1_left_corner[1] = 0.0;
        sub_grid_1_right_corner[0] = 0.0;
        sub_grid_1_right_corner[1] = free_height;
        sub_grid_2_left_corner[0] = 0.0;
        sub_grid_2_left_corner[1] = 0.0;
        sub_grid_2_right_corner[0] = plate_length;
        sub_grid_2_right_corner[1] = free_height;
        const bool colorize = true;

        std::vector<double> sub_grid_1_step_size_x(number_of_subdivisions_in_x_direction_free);
        for (int i=0;i<number_of_subdivisions_in_x_direction_free;++i){
            sub_grid_1_step_size_x[i] = free_length*((tanh(skewness_x_free*(1.0-(number_of_subdivisions_in_x_direction_free-1.0-i)/number_of_subdivisions_in_x_direction_free))-tanh(skewness_x_free*(1.0-(number_of_subdivisions_in_x_direction_free-i+0.0)/number_of_subdivisions_in_x_direction_free)))/tanh(skewness_x_free));
        }
        std::vector<double> step_size_y(number_of_subdivisions_in_y_direction);
        for(int i=0;i<number_of_subdivisions_in_y_direction;++i){
            step_size_y[i] = free_height*((tanh(skewness_y*(1.0-(i+0.0)/number_of_subdivisions_in_y_direction))-tanh(skewness_y*(1.0-(i+1.0)/number_of_subdivisions_in_y_direction)))/tanh(skewness_y));
        }
        std::vector<std::vector<double>> sub_grid_1_step_size;
        sub_grid_1_step_size.push_back(sub_grid_1_step_size_x);
        sub_grid_1_step_size.push_back(step_size_y);

        std::vector<double> sub_grid_2_step_size_x(number_of_subdivisions_in_x_direction_plate);
        for (int i=0;i<number_of_subdivisions_in_x_direction_plate;++i){
            sub_grid_2_step_size_x[i] = plate_length*((tanh(skewness_x_plate*(1.0-(i+0.0)/number_of_subdivisions_in_x_direction_plate))-tanh(skewness_x_plate*(1.0-(i+1.0)/number_of_subdivisions_in_x_direction_plate)))/tanh(skewness_x_plate));
        }
        std::vector<std::vector<double>> sub_grid_2_step_size;
        sub_grid_2_step_size.push_back(sub_grid_2_step_size_x);
        sub_grid_2_step_size.push_back(step_size_y);
        
        dealii::GridGenerator::subdivided_hyper_rectangle (*sub_grid_1, sub_grid_1_step_size, sub_grid_1_left_corner, sub_grid_1_right_corner, colorize);
        dealii::GridGenerator::subdivided_hyper_rectangle (*sub_grid_2, sub_grid_2_step_size, sub_grid_2_left_corner, sub_grid_2_right_corner, colorize);
        dealii::GridGenerator::merge_triangulations(*sub_grid_1,*sub_grid_2,*grid);

        for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    if (cell->face(face)->center()[0] == sub_grid_1_left_corner[0]){
                        cell->face(face)->set_boundary_id (1003);
                    } 
                    if (std::abs(cell->face(face)->center()[0] - sub_grid_2_right_corner[0]) <=1e-10){
                        cell->face(face)->set_boundary_id (1002);
                    } 
                    if (cell->face(face)->center()[1] == sub_grid_1_left_corner[1]){
                        if (cell->face(face)->center()[0]<0.0){
                            cell->face(face)->set_boundary_id (1006);
                        } else{
                            cell->face(face)->set_boundary_id (1001);
                        }
                    } 
                    if (std::abs(cell->face(face)->center()[1] - sub_grid_2_right_corner[1]) <=1e-10){
                        cell->face(face)->set_boundary_id (1005);
                    } 
                }
            }
            
        }
    }
}

#if PHILIP_DIM==1
    template void flat_plate_cube<PHILIP_DIM, dealii::Triangulation<PHILIP_DIM>> 
        (std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &grid,
         std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &sub_grid_1,
         std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &sub_grid_2,
         const double free_length,
         const double free_height,
         const double plate_length,
         const double skewness_x_free,
         const double skewness_x_plate,
         const double skewness_y,
         const int number_of_subdivisions_in_x_direction_free,
         const int number_of_subdivisions_in_x_direction_plate,
         const int number_of_subdivisions_in_y_direction);
#endif
#if PHILIP_DIM!=1
    template void flat_plate_cube<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> 
        (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid,
         std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &sub_grid_1,
         std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &sub_grid_2, 
         const double free_length,
         const double free_height,
         const double plate_length,
         const double skewness_x_free,
         const double skewness_x_plate,
         const double skewness_y,
         const int number_of_subdivisions_in_x_direction_free,
         const int number_of_subdivisions_in_x_direction_plate,
         const int number_of_subdivisions_in_y_direction);
#endif

} // namespace Grids
} // namespace PHiLiP