#include "gaussian_bump.h"
#include "mesh/grids/gaussian_bump.h"

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

    const unsigned int number_of_refinements = this->all_param.grid_refinement_study_param.num_refinements;
    const double channel_length = this->all_param.mesh_generation_param.channel_length;
    const double channel_height = this->all_param.mesh_generation_param.channel_height;

    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[0] = this->all_param.mesh_generation_param.number_of_subdivisions_in_x_direction;
    n_subdivisions[1] = this->all_param.mesh_generation_param.number_of_subdivisions_in_y_direction;
    // n_subdivisions[2] = his->all_param.mesh_generation_param.number_of_subdivisions_in_z_direction;

    Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void GaussianBump<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Channel length: " << this->all_param.mesh_generation_param.channel_length << std::endl;
    this->pcout << "- - Channel height: " << this->all_param.mesh_generation_param.channel_height << std::endl;
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

#if PHILIP_DIM==2
template class GaussianBump<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}