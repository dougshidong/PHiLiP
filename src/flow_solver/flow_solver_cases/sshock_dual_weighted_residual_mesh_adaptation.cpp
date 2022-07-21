#include "sshock_dual_weighted_residual_mesh_adaptation.h"
#include <deal.II/grid/grid_generator.h>

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
SshockMeshAdaptation<dim, nstate>::SshockMeshAdaptation(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<Triangulation> SshockMeshAdaptation<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator); // Mesh smoothing is set to none by default.
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void SshockMeshAdaptation<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Do nothing.
}

#if PHILIP_DIM==2
    template class SshockMeshAdaptation<PHILIP_DIM, 1>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace
