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
                                                      const bool is_oblique_)
        : PeriodicTurbulence<dim, nstate>(parameters_input)
        , is_oblique(is_oblique_)
        , do_use_stretched_mesh(this->all_param.flow_solver_param.do_use_stretched_mesh)
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> DipoleWallCollision<dim,nstate>::generate_grid() const
{
    if(this->do_use_stretched_mesh) 
        return this->generate_grid_stretched();
    else 
        return this->generate_grid_uniform();
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> DipoleWallCollision<dim,nstate>::generate_grid_uniform() const
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

template <int dim, int nstate>
std::vector<double> DipoleWallCollision<dim,nstate>::get_mesh_step_size_stretched(
    const int number_of_cells_, 
    const double domain_length_) const 
{
    // Set the parameters for the DWC flow case
    const int number_of_cells_y_direction = number_of_cells_;
    const double domain_length_y = domain_length_;
    const double pi_val = 3.141592653589793238;
    // ------------------------------------------------
    // CODE BELOW TAKEN FROM CHANNEL FLOW CLASS
    // ------------------------------------------------
    // - get stretched spacing for y-direction to capture boundary layer
    const int number_of_edges_y_direction = number_of_cells_y_direction+1;
    std::vector<double> element_edges_y_direction(number_of_edges_y_direction);
    /**
     * Reference: C. CARTON DE WIARTET. AL, "Implicit LES of free and wall-bounded turbulent flows based onthe discontinuous Galerkin/symmetric interior penalty method", 2015.
     **/
    const double num_cells_y = (double)number_of_cells_y_direction;
    const double uniform_spacing = domain_length_y/num_cells_y;
    for (int j=0; j<(number_of_cells_y_direction/2+1); j++) {
        element_edges_y_direction[j] = 1.0 - cos(pi_val*((double)j)*uniform_spacing/2.0);
        element_edges_y_direction[number_of_cells_y_direction-j] = domain_length_y-element_edges_y_direction[j];
    }
    // - compute the step size in y-direction as the difference between element edges in y-direction
    std::vector<double> step_size_y_direction(number_of_cells_y_direction);
    for (int j=0; j<number_of_cells_y_direction; j++) {
        step_size_y_direction[j] = element_edges_y_direction[j+1] - element_edges_y_direction[j];
    }
    return step_size_y_direction;
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> DipoleWallCollision<dim,nstate>::generate_grid_stretched() const
{
    // Set the parameters for the DWC flow case
    const double domain_length_x = this->domain_right-this->domain_left;
    const double domain_length_y = this->domain_right-this->domain_left;
    const int number_of_cells_x_direction = this->number_of_cells_per_direction;
    const int number_of_cells_y_direction = this->number_of_cells_per_direction;
    // ------------------------------------------------
    // CODE BELOW TAKEN FROM CHANNEL FLOW CLASS -- Commented z-component stuff since 2D here
    // ------------------------------------------------

    // // uncomment this to use the gmsh reader
    // // Dummy triangulation
    // // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
    // const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    // const bool use_mesh_smoothing = false;
    // const int grid_order = 0;
    // std::shared_ptr<HighOrderGrid<dim,double>> mesh = read_gmsh<dim, dim> (mesh_filename, grid_order, use_mesh_smoothing);
    // return mesh->triangulation;

    // define domain to be centered about x, y, and z axes
    const dealii::Point<dim> p1(-0.5*domain_length_x, -0.5*domain_length_y/*, -0.5*domain_length_z*/);
    const dealii::Point<dim> p2(0.5*domain_length_x, 0.5*domain_length_y/*, 0.5*domain_length_z*/);

    // get step size for each cell
    // - uniform spacing in x and z
    // const double uniform_spacing_x = domain_length_x/double(number_of_cells_x_direction);
    const double uniform_spacing_y = domain_length_x/double(number_of_cells_y_direction);
    /*const double uniform_spacing_z = domain_length_z/double(number_of_cells_z_direction);*/
    // - get stretched spacing for possible wall-normal directions
    std::vector<double> step_size_x_direction = get_mesh_step_size_stretched(number_of_cells_x_direction,domain_length_x);
    std::vector<double> step_size_y_direction = get_mesh_step_size_stretched(number_of_cells_y_direction,domain_length_y);

    std::vector<std::vector<double> > step_sizes(dim);
    // x-direction (wall normal)
    for (int i=0; i<number_of_cells_x_direction; i++) {
        step_sizes[0].push_back(step_size_x_direction[i]);
    }
    // y-direction
    for (int j=0; j<number_of_cells_y_direction; j++) {
        if(this->is_oblique) step_sizes[1].push_back(step_size_y_direction[j]); // wall-normal for oblique case
        else step_sizes[1].push_back(uniform_spacing_y);
    }
    /*
    // z-direction
    for (int k=0; k<number_of_cells_z_direction; k++) {
        step_sizes[2].push_back(uniform_spacing_z);
    }*/

    // generate grid usign dealii
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator);
    const bool colorize = true;
    dealii::GridGenerator::subdivided_hyper_rectangle(*grid, step_sizes, p1, p2, colorize);

    if(!this->is_oblique) {
        // assign periodic boundary conditions in x and z
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        // dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs); // x-direction
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs); // y-direction
        /*dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs); // z-direction*/
        grid->add_periodicity(matched_pairs);
    }

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