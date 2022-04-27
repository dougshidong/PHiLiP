#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include "initial_condition_base.h"
#include "dg/dg.h"

namespace PHiLiP{

//Constructor
template<int dim, typename real>
InitialConditionBase<dim,real>::InitialConditionBase(
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input)
        : all_parameters(parameters_input)
        , nstate(nstate_input)
        , dg(dg_input)
        , initial_condition_function(InitialConditionFactory<dim,double>::create_InitialConditionFunction(parameters_input, nstate_input))
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{

    pcout<<"Setting up Initial Condition"<<std::endl;
    if(all_parameters->flow_solver_param.interpolate_initial_condition)
        interpolate_initial_condition(dg);
    else
        project_initial_condition(dg);
}

//Destructor
template<int dim, typename real>
InitialConditionBase<dim,real>::~InitialConditionBase ()
{}

template<int dim, typename real>
void InitialConditionBase<dim,real>::interpolate_initial_condition(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler,*initial_condition_function,solution_no_ghost);
    dg->solution = solution_no_ghost;
}

template<int dim, typename real>
void InitialConditionBase<dim,real>::project_initial_condition(
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

template class InitialConditionBase<PHILIP_DIM, double>;

}//end of namespace PHILIP
