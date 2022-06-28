#include "gaussian_bump.h"
#include <iostream>
#include <stdlib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include "mesh/grids/gaussian_bump.h"
#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP{
namespace Tests{

template <int dim, int nstate>
GaussianBump<dim, nstate>::GaussianBump(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
    , TestsBase(parameters_input)
{
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> GaussianBump<dim,nstate>::generate_grid() const {
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM != 1
            this->FlowSolverCaseBase<dim, nstate>::mpi_communicator
#endif
    );

    const unsigned int n_grids = this->all_param.manufactured_convergence_study_param.number_of_grids;
    const unsigned int number_of_refinements = this->all_param.grid_refinement_study_param.num_refinements;

    const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);
    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[1] = n_1d_cells[0]; // y-direction
    n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

    const double channel_length = 3.0;
    const double channel_height = 0.8;

    Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);
    grid->refine_global(number_of_refinements);

    return grid;}



#if PHILIP_DIM==2
template class GaussianBump<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}