#include "mesh_adaptation.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshAdaptation<dim,real,MeshType>::MeshAdaptation(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
    , critical_residual(dg->all_parameters->mesh_adaptation_param.critical_residual_val)
    , total_refinement_cycles(dg->all_parameters->mesh_adaptation_param.total_refinement_steps)
    , current_refinement_cycle(0)
    , refinement_fraction(dg->all_parameters->mesh_adaptation_param.refinement_fraction)
    , coarsening_fraction(dg->all_parameters->mesh_adaptation_param.coarsening_fraction)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
        mesh_error = MeshErrorFactory<dim, 5, real, MeshType> :: create_mesh_error(dg);
    }


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::adapt_mesh()
{
    cellwise_errors = mesh_error->compute_cellwise_errors();

    fixed_fraction_isotropic_refinement_and_coarsening();
    current_refinement_cycle++;
    pcout<<"Refined"<<std::endl;
}


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::fixed_fraction_isotropic_refinement_and_coarsening()
{
    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    dg->high_order_grid->prepare_for_coarsening_and_refinement();

    if constexpr(dim == 1 || !std::is_same<MeshType, dealii::parallel::distributed::Triangulation<dim>>::value) 
    {
        dealii::GridRefinement::refine_and_coarsen_fixed_number(*(dg->high_order_grid->triangulation),
                                                                cellwise_errors,
                                                                refinement_fraction,
                                                                coarsening_fraction);
    } 
    else 
    {
        dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(*(dg->high_order_grid->triangulation),
                                                                                        cellwise_errors,
                                                                                        refinement_fraction,
                                                                                        coarsening_fraction);
    }

    dg->high_order_grid->triangulation->execute_coarsening_and_refinement();
    dg->high_order_grid->execute_coarsening_and_refinement();
    
    dg->allocate_system ();
    dg->solution.zero_out_ghosts();
    solution_transfer.interpolate(dg->solution);
    dg->solution.update_ghost_values();
    dg->assemble_residual ();
}

template class MeshAdaptation<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
