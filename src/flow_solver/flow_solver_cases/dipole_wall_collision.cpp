#include "dipole_wall_collision.h"
#include <deal.II/dofs/dof_tools.h>
// #include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/base/tensor.h>
#include "math.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
// #include "mesh/gmsh_reader.hpp" // uncomment this to use the gmsh reader

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// DIPOLE WALL COLLISION CLASS
//=========================================================
template <int dim, int nstate>
DipoleWallCollision<dim, nstate>::DipoleWallCollision(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                      const bool is_oblique)
        : PeriodicTurbulence<dim, nstate>(parameters_input)
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> DipoleWallCollision<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );

    // Get equivalent number of refinements
    const int number_of_refinements = log(this->number_of_cells_per_direction)/log(2);

    // Check that number_of_cells_per_direction is a power of 2 if number_of_refinements is non-zero
    if(number_of_refinements >= 0){
        int val_check = this->number_of_cells_per_direction;
        while(val_check > 1) {
            if(val_check % 2 == 0) val_check /= 2;
            else{
                std::cout << "ERROR: number_of_cells_per_direction is not a power of 2. " 
                          << "Current value is " << this->number_of_cells_per_direction << ". "
                          << "Change value of number_of_grid_elements_per_dimension in .prm file." << std::endl;
                std::abort();
            }
        }
    }
    
    // Definition for each type of grid
    std::string grid_type_string;
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, this->domain_left, this->domain_right, colorize);
    if constexpr(dim==2) {
        if(!this->is_oblique) {
            // grid_type_string = "Doubly periodic square.";
            std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
            // dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs); // x-direction
            dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs); // y-direction
            // dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs); // z-direction
            grid->add_periodicity(matched_pairs);
        }
    }
    grid->refine_global(number_of_refinements);
    // assign wall boundary conditions
    for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if(this->is_oblique) {
                    if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                }
                if (current_id == 0 || current_id == 1) cell->face(face)->set_boundary_id (1001); // Left and right wall
                // could simply introduce different boundary id if using a wall model
            }
        }
    }

    return grid;
}

//=========================================================
// DIPOLE WALL COLLISION CLASS -- OBLIQUE
//=========================================================
template <int dim, int nstate>
DipoleWallCollision_Oblique<dim, nstate>::DipoleWallCollision_Oblique(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : DipoleWallCollision<dim, nstate>(parameters_input,true)
{ }

#if PHILIP_DIM==2
template class DipoleWallCollision <PHILIP_DIM,PHILIP_DIM+2>;
template class DipoleWallCollision_Oblique <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace