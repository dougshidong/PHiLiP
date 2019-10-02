#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "high_order_grid.h"
namespace PHiLiP {

template <int dim, typename real, typename VectorType , typename DoFHandlerType>
HighOrderGrid<dim,real,VectorType,DoFHandlerType>::HighOrderGrid(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int max_degree,
        dealii::Triangulation<dim> *const triangulation_input)
    : all_parameters(parameters_input)
    , max_degree(max_degree)
    , triangulation(triangulation_input)
    , dof_handler_grid(*triangulation)
    , fe_q(max_degree) // The grid must be at least p1. A p0 solution required a p1 grid.
    , fe_system(dealii::FESystem<dim>(fe_q,dim)) // The grid must be at least p1. A p0 solution required a p1 grid.
    , solution_transfer(dof_handler_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    allocate();
}

template <int dim, typename real, typename VectorType , typename DoFHandlerType>
void 
HighOrderGrid<dim,real,VectorType,DoFHandlerType>::allocate() 
{
    dof_handler_grid.initialize(*triangulation, fe_system);
    dof_handler_grid.distribute_dofs(fe_system);

    locally_owned_dofs_grid = dof_handler_grid.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_grid, ghost_dofs_grid);
    locally_relevant_dofs_grid = ghost_dofs_grid;
    ghost_dofs_grid.subtract_set(locally_owned_dofs_grid);
    nodes.reinit(locally_owned_dofs_grid, ghost_dofs_grid, mpi_communicator);
}

template <int dim, typename real, typename VectorType , typename DoFHandlerType>
dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> 
HighOrderGrid<dim,real,VectorType,DoFHandlerType>::get_MappingFEField() {
    const dealii::ComponentMask mask(dim, true);
    dealii::VectorTools::get_position_vector(dof_handler_grid, nodes, mask);

    dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> mapping(dof_handler_grid,nodes,mask);

    return mapping;
}


template <int dim, typename real, typename VectorType , typename DoFHandlerType>
void HighOrderGrid<dim,real,VectorType,DoFHandlerType>::prepare_for_coarsening_and_refinement() {

    old_nodes = nodes;
    old_nodes.update_ghost_values();
    solution_transfer.prepare_for_coarsening_and_refinement(old_nodes);
}

template <int dim, typename real, typename VectorType , typename DoFHandlerType>
void HighOrderGrid<dim,real,VectorType,DoFHandlerType>::execute_coarsening_and_refinement() {
    allocate();
    solution_transfer.interpolate(nodes);
    nodes.update_ghost_values();
}


//template class HighOrderGrid<PHILIP_DIM, double>;
template class HighOrderGrid<PHILIP_DIM, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<PHILIP_DIM>>;
} // namespace PHiLiP
