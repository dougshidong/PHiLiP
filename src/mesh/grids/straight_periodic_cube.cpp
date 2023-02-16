#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <stdlib.h>
#include <iostream>

#include "straight_periodic_cube.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void straight_periodic_cube(std::shared_ptr<TriangulationType> &grid,
                            const double domain_left,
                            const double domain_right,
                            const int number_of_cells_per_direction)
{
    // Get equivalent number of refinements
    const int number_of_refinements = log(number_of_cells_per_direction)/log(2);

    // Check that number_of_cells_per_direction is a power of 2 if number_of_refinements is non-zero
    if(number_of_refinements >= 0){
        int val_check = number_of_cells_per_direction;
        while(val_check > 1) {
            if(val_check % 2 == 0) val_check /= 2;
            else{
                std::cout << "ERROR: number_of_cells_per_direction is not a power of 2. " 
                          << "Current value is " << number_of_cells_per_direction << ". "
                          << "Change value of number_of_grid_elements_per_dimension in .prm file." << std::endl;
                std::abort();
            }
        }
    }
    
    // Definition for each type of grid
    std::string grid_type_string;
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    if constexpr(dim == 1){
        grid_type_string = "Periodic 1D domain.";
        std::vector<dealii::GridTools::PeriodicFacePair<typename TriangulationType::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }else if constexpr(dim==2) {
        grid_type_string = "Doubly periodic square.";
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }else if constexpr(dim==3) {
        grid_type_string = "Triply periodic cube.";
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }
    grid->refine_global(number_of_refinements);
}

#if PHILIP_DIM==1
    template void straight_periodic_cube<PHILIP_DIM, dealii::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif
#if PHILIP_DIM!=1
    template void straight_periodic_cube<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif

} // namespace Grids
} // namespace PHiLiP
