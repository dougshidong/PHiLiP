#include "non_periodic_cube_flow.h"
#include <deal.II/grid/grid_generator.h>

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nspecies, int nstate>
NonPeriodicCubeFlow<dim, nspecies, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nspecies, nstate>(parameters_input)
{}

template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim,nspecies,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator); // Mesh smoothing is set to none by default.
    
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nspecies, int nstate>
void NonPeriodicCubeFlow<dim,nspecies,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Do nothing for now.
}

#if PHILIP_DIM>1
    template class NonPeriodicCubeFlow<PHILIP_DIM, 1, 1>;
    template class NonPeriodicCubeFlow <PHILIP_DIM,PHILIP_SPECIES, PHILIP_DIM+2+PHILIP_SPECIES-1>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace