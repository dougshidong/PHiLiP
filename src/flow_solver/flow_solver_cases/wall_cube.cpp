#include "wall_cube.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_bounded_cube.hpp"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include "mesh/grids/naca_airfoil_grid.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// DISTANCE EVALUATION IN BOUNDED CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
WallCube<dim, nstate>::WallCube(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> WallCube<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );

    //Grids::straight_bounded_cube<dim,Triangulation>(grid, domain_left, domain_right, number_of_cells_per_direction);

#if PHILIP_DIM==2
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id      = "0012";
    airfoil_data.airfoil_length = 1.0;
    airfoil_data.height         = 2.0;
    airfoil_data.length_b2      = 2.0;
    airfoil_data.incline_factor = 0.0;
    airfoil_data.bias_factor    = 1; 
    airfoil_data.refinements    = 0;

    const int n_subdivisions_0 = 60;
    const int n_subdivisions_1 = 40;
    const int n_cells_airfoil = n_subdivisions_0;
    const int n_cells_downstream = n_subdivisions_0/2;
    airfoil_data.n_subdivision_x_0 = n_cells_airfoil;
    airfoil_data.n_subdivision_x_1 = n_cells_airfoil / 20;
    airfoil_data.n_subdivision_x_2 = n_cells_downstream;
    airfoil_data.n_subdivision_y = n_subdivisions_1;
    airfoil_data.airfoil_sampling_factor = 3; 

    //Grids::naca_airfoil(*grid,airfoil_data);
    dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
    //grid->refine_global();
    // Assign a manifold to have curved geometry
    unsigned int manifold_id = 0;
    grid->reset_all_manifolds();
    grid->set_all_manifold_ids(manifold_id);
    // Set Flat manifold on the domain, but not on the boundary.
    grid->set_manifold(manifold_id, dealii::FlatManifold<2>());

    manifold_id = 1;
    bool is_upper = true;
    const Grids::NACAManifold<2,1> upper_naca(airfoil_data.naca_id, is_upper);
    grid->set_all_manifold_ids_on_boundary(2,manifold_id); // upper airfoil side
    grid->set_manifold(manifold_id, upper_naca);

    is_upper = false;
    const Grids::NACAManifold<2,1> lower_naca(airfoil_data.naca_id, is_upper);
    manifold_id = 2;
    grid->set_all_manifold_ids_on_boundary(3,manifold_id); // lower airfoil side
    grid->set_manifold(manifold_id, lower_naca); 

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                    cell->face(face)->set_boundary_id (1005); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1001); // wall bc
                }
            }
        }
    }
#endif
    return grid;
}

template <int dim, int nstate>
void WallCube<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    const std::string grid_type_string = "straight_bounded_cube";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << this->domain_left << std::endl;
    this->pcout << "- - Domain right: " << this->domain_right << std::endl;
    this->pcout << "- - Number of cells in each direction: " << this->number_of_cells_per_direction << std::endl;
    if constexpr(dim==2) this->pcout << "- - Domain area: " << this->domain_size << std::endl;
    //if constexpr(dim==3) this->pcout << "- - Domain volume: " << this->domain_size << std::endl;
}

template class WallCube <PHILIP_DIM,1>;

} // FlowSolver namespace
} // PHiLiP namespace