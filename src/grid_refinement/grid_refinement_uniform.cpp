#include <deal.II/dofs/dof_tools.h>

#include "parameters/parameters_grid_refinement.h"

#include "dg/dg.h"
#include "mesh/high_order_grid.h"

#include "grid_refinement_uniform.h"

namespace PHiLiP {

namespace GridRefinement {

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // setting up the solution transfer
    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs = this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);

    dealii::LinearAlgebra::distributed::Vector<real> solution_old(this->dg->solution);
    solution_old.update_ghost_values();

    dealii::parallel::distributed::SolutionTransfer< 
        dim, dealii::LinearAlgebra::distributed::Vector<real>, dealii::DoFHandler<dim> 
        > solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_old);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    this->tria->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    // transfering the solution from solution_old
    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();
    solution_transfer.interpolate(this->dg->solution);
    this->dg->solution.update_ghost_values();

    // increase the count
    this->iteration++;
}

// functions for the refinement calls for each of the classes
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_h()
{
    this->tria->set_all_refine_flags();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_p()
{
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index()+1 <= this->dg->max_degree)
            cell->set_future_fe_index(cell->active_fe_index()+1);
    
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_hp()
{
    refine_grid_h();
    refine_grid_p();
}

template <int dim, int nstate, typename real, typename MeshType>
std::vector< std::pair<dealii::Vector<real>, std::string> > GridRefinement_Uniform<dim,nstate,real,MeshType>::output_results_vtk_method()
{
    // nothing special to do here
    std::vector< std::pair<dealii::Vector<real>, std::string> > data_out_vector;

    return data_out_vector;
}

// dealii::Triangulation<PHILIP_DIM>
template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace GridRefinement

} // namespace PHiLiP
