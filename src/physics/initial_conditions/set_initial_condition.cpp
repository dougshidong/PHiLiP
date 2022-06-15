#include "set_initial_condition.h"
#include <deal.II/numerics/vector_tools.h>

namespace PHiLiP{

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::set_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input)
{
    // Apply initial condition depending on the application type
    const bool interpolate_initial_condition = parameters_input->flow_solver_param.interpolate_initial_condition;
    if(interpolate_initial_condition == true) {
        // for non-curvilinear
        SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(initial_condition_function_input, dg_input);
    } else {
        // for curvilinear
        SetInitialCondition<dim,nstate,real>::project_initial_condition(initial_condition_function_input, dg_input);
    }
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler,*initial_condition_function,solution_no_ghost);
    dg->solution = solution_no_ghost;
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::project_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    //Note that for curvilinear, can't use dealii interpolate since it doesn't project at the correct order.
    //Thus we interpolate it directly.
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->operators->fe_collection_basis, dg->operators->volume_quadrature_collection, 
                                dealii::update_quadrature_points);
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_quad_pts = dg->operators->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_dofs_cell = dg->operators->fe_collection_basis[poly_degree].dofs_per_cell;
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->operators->fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            dg->solution[current_dofs_indices[idof]] = 0.0;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));
                double exact_value = initial_condition_function->value(qpoint, istate);
                //project the solution from quad points to dofs
                dg->solution[current_dofs_indices[idof]] += dg->operators->vol_projection_operator[poly_degree][idof][iquad] * exact_value; 
            }   
        }
    }
}

template class SetInitialCondition<PHILIP_DIM, 1, double>;
template class SetInitialCondition<PHILIP_DIM, 2, double>;
template class SetInitialCondition<PHILIP_DIM, 3, double>;
template class SetInitialCondition<PHILIP_DIM, 4, double>;
template class SetInitialCondition<PHILIP_DIM, 5, double>;

}//end of namespace PHILIP
