#include "mesh_adaptation.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshAdaptation<dim,real,MeshType>::MeshAdaptation(double critical_res_input, int total_ref_cycle, double refine_frac, double coarsen_frac)
    : critical_residual(critical_res_input)
    , total_refinement_cycles(total_ref_cycle)
    , refinement_fraction(refine_frac)
    , coarsening_fraction(coarsen_frac)
    , current_refinement_cycle(0)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
        mesh_error = std::make_unique<ResidualErrorEstimate<dim, real, MeshType>>();
    }


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::adapt_mesh(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
{
    if (!refine_mesh)
    {
        return;
    }

    if (total_refinement_cycles == current_refinement_cycle)
    {
        return;
    }

    cellwise_errors = mesh_error->compute_cellwise_errors(dg);

    fixed_fraction_isotropic_refinement_and_coarsening(dg);
    current_refinement_cycle++;
    pcout<<"Refined"<<std::endl;
}


template <int dim, typename real, typename MeshType>
void MeshAdaptation<dim,real,MeshType>::fixed_fraction_isotropic_refinement_and_coarsening(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
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
