#include "limiter_conv_tests.h"
#include <iostream>
#include <stdlib.h>
#include "mesh/gmsh_reader.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>


namespace PHiLiP{
namespace FlowSolver{

template <int dim, int nstate>
LimiterConvTests<dim, nstate>::LimiterConvTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<Triangulation> LimiterConvTests<dim,nstate>::generate_grid() const
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    double left = this->all_param.flow_solver_param.grid_left_bound;
    double right = this->all_param.flow_solver_param.grid_right_bound;
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;

    //straight grid setup
    dealii::GridGenerator::hyper_cube(*grid, left, right, true);
#if PHILIP_DIM==1
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
    grid->add_periodicity(matched_pairs);
#else
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
    if (dim >= 2) dealii::GridTools::collect_periodic_faces(*grid, 2, 3, 1, matched_pairs);
    if (dim >= 3) dealii::GridTools::collect_periodic_faces(*grid, 4, 5, 2, matched_pairs);
    grid->add_periodicity(matched_pairs);
#endif
    grid->refine_global(number_of_refinements);
    std::cout << "Grid generated and refined" << std::endl;

    return grid;
    
    // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
}

template <int dim, int nstate>
void LimiterConvTests<dim, nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

#if PHILIP_DIM==1
    template class LimiterConvTests<PHILIP_DIM, PHILIP_DIM>;
#elif PHILIP_DIM==2
    template class LimiterConvTests<PHILIP_DIM, PHILIP_DIM>;
    template class LimiterConvTests<PHILIP_DIM, PHILIP_DIM+2>;
    template class LimiterConvTests<PHILIP_DIM, 1>;
#endif

}
}