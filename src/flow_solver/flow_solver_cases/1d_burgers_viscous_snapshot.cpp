#include "1d_burgers_viscous_snapshot.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>
#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
BurgersViscousSnapshot<dim, nstate>::BurgersViscousSnapshot(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_refinements(this->all_param.grid_refinement_study_param.num_refinements)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
{
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> BurgersViscousSnapshot<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void BurgersViscousSnapshot<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Display the information about the grid
    this->pcout << "\n- GRID INFORMATION:" << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << domain_left << std::endl;
    this->pcout << "- - Domain right: " << domain_right << std::endl;
    this->pcout << "- - Number of refinements:  " << number_of_refinements << std::endl;
}

#if PHILIP_DIM==1
template class BurgersViscousSnapshot<PHILIP_DIM,PHILIP_DIM>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace
