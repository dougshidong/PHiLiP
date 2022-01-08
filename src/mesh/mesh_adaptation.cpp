#include "mesh_adaptation.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshAdaptation<dim,real,MeshType>::MeshAdaptation()
    : critical_residual(1.0e-5)
    , total_refinement_cycles(15)
    , current_refinement_cycle(0)
    {}

template <int dim, typename real, typename MeshType>
int MeshAdaptation<dim,real,MeshType>::adapt_mesh(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
{
    if (!refine_mesh)
    {
        return 0;
    }

    if (total_refinement_cycles == current_refinement_cycle)
    {
        return 0;
    }

    cellwise_errors.reinit(0);
    cellwise_errors.reinit(dg->high_order_grid->triangulation->n_active_cells());
    compute_cellwise_errors(dg);

    fixed_fraction_isotropic_refinement_and_coarsening(dg);
    current_refinement_cycle++;

    return 0;
}

template <int dim, typename real, typename MeshType>
int MeshAdaptation<dim,real,MeshType>::compute_cellwise_errors(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
{
    compute_max_cellwise_residuals(dg); // Future extension: Error depends on parameters input (i.e. this function computes residual or goal-oriented error).
    return 0;
}


template <int dim, typename real, typename MeshType>
int MeshAdaptation<dim,real,MeshType>::compute_max_cellwise_residuals(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
{
    std::vector<dealii::types::global_dof_index> dofs_indices;
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
         if (!cell->is_locally_owned()) 
         continue;
 
         const int i_fele = cell->active_fe_index();
         const dealii::FESystem<dim,dim> &fe_ref = dg->fe_collection[i_fele];
         const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();
         dofs_indices.resize(n_dofs_cell);
         cell->get_dof_indices (dofs_indices);
         double max_residual = 0;
         for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) 
         {
             const unsigned int index = dofs_indices[idof];
             const unsigned int istate = fe_ref.system_to_component_index(idof).first;
             if (istate == dim+2-1) 
             {
                 const double res = std::abs(dg->right_hand_side[index]);
                 if (res > max_residual) max_residual = res;
             }
         }
         cellwise_errors[cell->active_cell_index()] = max_residual;
     }

    return 0;
}


template <int dim, typename real, typename MeshType>
int MeshAdaptation<dim,real,MeshType>::fixed_fraction_isotropic_refinement_and_coarsening(std::shared_ptr< DGBase<dim, real, MeshType> > dg)
{
    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    dg->high_order_grid->prepare_for_coarsening_and_refinement();

    double refinement_fraction = 0.05;
    double coarsening_fraction = 0.01;
 
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


    return 0;
}

template class MeshAdaptation<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshAdaptation<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
