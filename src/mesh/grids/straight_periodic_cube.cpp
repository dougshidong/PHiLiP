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

    // Definition for each type of grid
    std::string grid_type_string;
    if(dim==3) {
        grid_type_string = "Triply periodic cube.";
        const bool colorize = true;
        dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
        grid->refine_global(number_of_refinements);
    }
    else {
    	std::cout << " ERROR: straight_periodic_cube() cannot be called for dim!=3 " << std::endl;
    	std::abort();
    }
}

template void straight_periodic_cube<3, dealii::parallel::distributed::Triangulation<3>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<3>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);

} // namespace Grids
} // namespace PHiLiP