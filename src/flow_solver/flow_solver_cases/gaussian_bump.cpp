#include "gaussian_bump.h"
#include "mesh/grids/gaussian_bump.h"
#include "mesh/gmsh_reader.hpp"

#include <iostream>
#include <stdlib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>


namespace PHiLiP{
namespace FlowSolver{

template <int dim, int nstate>
GaussianBump<dim, nstate>::GaussianBump(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<Triangulation> GaussianBump<dim,nstate>::generate_grid() const 
{
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation>(
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    if constexpr(dim==2) {
        const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
        const double channel_length = this->all_param.flow_solver_param.channel_length;
        const double channel_height = this->all_param.flow_solver_param.channel_height;
        const double bump_height = this->all_param.flow_solver_param.bump_height;

        std::vector<unsigned int> n_subdivisions(dim);
        n_subdivisions[0] = this->all_param.flow_solver_param.number_of_subdivisions_in_x_direction;
        n_subdivisions[1] = this->all_param.flow_solver_param.number_of_subdivisions_in_y_direction;
        // n_subdivisions[2] = this->all_param.flow_solver_param.grid.gaussian_bump.number_of_subdivisions_in_z_direction;

        Grids::gaussian_bump<dim>(*grid, n_subdivisions, channel_length, channel_height, bump_height);
        grid->refine_global(number_of_refinements);
        return grid;
    } 
    if constexpr(dim==3) {
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        std::shared_ptr<HighOrderGrid<dim,double>> gaussian_bump_mesh = read_gmsh<dim, dim> (mesh_filename);
        return gaussian_bump_mesh->triangulation;
    }
    
    // TO DO: Do nothing for 3D, set_high_order_grid() will accomplish this.
}

template <int dim, int nstate>
void GaussianBump<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> /*dg*/) const
{
    // if constexpr(dim==3) {
    //     const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    //     std::shared_ptr<HighOrderGrid<dim,double>> gaussian_bump_mesh = read_gmsh<dim, dim> (mesh_filename);
    //     dg->set_high_order_grid(gaussian_bump_mesh);
    //     for (int i=0; i<this->all_param.flow_solver_param.number_of_mesh_refinements; ++i) {
    //         dg->high_order_grid->refine_global();
    //     }
    // }
}

template <int dim, int nstate>
void GaussianBump<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    if constexpr(dim==2) {
        this->pcout << "- - Gaussian bump parameters: " << std::endl;
        this->pcout << "- - - Channel length: " << this->all_param.flow_solver_param.channel_length << std::endl;
        this->pcout << "- - - Channel height: " << this->all_param.flow_solver_param.channel_height << std::endl;
    }
    const dealii::Point<dim> dummy_point;
    this->pcout << "- - Initialization: " << std::endl;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

#if PHILIP_DIM!=1
    template class GaussianBump<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}