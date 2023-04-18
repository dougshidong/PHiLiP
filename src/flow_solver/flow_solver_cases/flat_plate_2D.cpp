#include "flat_plate_2D.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/flat_plate_cube.hpp"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include "mesh/grids/naca_airfoil_grid.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// DISTANCE EVALUATION IN BOUNDED CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
FlatPlate2D<dim, nstate>::FlatPlate2D(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , free_length(this->all_param.flow_solver_param.free_length)
        , free_height(this->all_param.flow_solver_param.free_height)
        , plate_length(this->all_param.flow_solver_param.plate_length)
        , skewness_x_free(this->all_param.flow_solver_param.skewness_x_free)
        , skewness_x_plate(this->all_param.flow_solver_param.skewness_x_plate)
        , skewness_y(this->all_param.flow_solver_param.skewness_y)
        , number_of_subdivisions_in_x_direction_free(this->all_param.flow_solver_param.number_of_subdivisions_in_x_direction_free)
        , number_of_subdivisions_in_x_direction_plate(this->all_param.flow_solver_param.number_of_subdivisions_in_x_direction_plate)
        , number_of_subdivisions_in_y_direction(this->all_param.flow_solver_param.number_of_subdivisions_in_y_direction)
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> FlatPlate2D<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );
    std::shared_ptr<Triangulation> sub_grid_1 = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );
    std::shared_ptr<Triangulation> sub_grid_2 = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );

    Grids::flat_plate_cube<dim,Triangulation>(grid, sub_grid_1, sub_grid_2, free_length, free_height, plate_length, skewness_x_free, skewness_x_plate, skewness_y, number_of_subdivisions_in_x_direction_free, number_of_subdivisions_in_x_direction_plate, number_of_subdivisions_in_y_direction);

    return grid;
}

template <int dim, int nstate>
void FlatPlate2D<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    const std::string grid_type_string = "flat_plate_cube";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - free length is : " << this->free_length << std::endl;
    this->pcout << "- - free height is : " << this->free_height << std::endl;
    this->pcout << "- - plate length is : " << this->plate_length << std::endl;
    this->pcout << "- - skewness of cells for free area in x direction: " << this->skewness_x_free << std::endl;
    this->pcout << "- - skewness of cells for plate area in x direction: " << this->skewness_x_plate << std::endl;
    this->pcout << "- - skewness of cells in y direction: " << this->skewness_y << std::endl;
    this->pcout << "- - Number of cells for free area in x direction: " << this->number_of_subdivisions_in_x_direction_free << std::endl;
    this->pcout << "- - Number of cells for plate area in x direction: " << this->number_of_subdivisions_in_x_direction_plate << std::endl;
    this->pcout << "- - Number of cells in y direction: " << this->number_of_subdivisions_in_y_direction << std::endl;
}

template class FlatPlate2D <PHILIP_DIM,1>;
template class FlatPlate2D <PHILIP_DIM,PHILIP_DIM+2>;
template class FlatPlate2D <PHILIP_DIM,PHILIP_DIM+3>;

} // FlowSolver namespace
} // PHiLiP namespace