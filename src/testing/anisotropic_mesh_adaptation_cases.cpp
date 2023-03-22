#include <stdlib.h>
#include <iostream>
#include "anisotropic_mesh_adaptation_cases.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"
#include "mesh/mesh_adaptation/fe_values_shape_hessian.h"
#include <deal.II/grid/grid_in.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AnisotropicMeshAdaptationCases<dim, nstate> :: AnisotropicMeshAdaptationCases(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    const bool use_goal_oriented_approach = true;
    const double complexity = 50;
    double normLp = 2.0;
    if(use_goal_oriented_approach) {normLp = 1.0;}

    std::unique_ptr<AnisotropicMeshAdaptation<dim, nstate, double>> anisotropic_mesh_adaptation =
                        std::make_unique<AnisotropicMeshAdaptation<dim, nstate, double>> (flow_solver->dg, normLp, complexity, use_goal_oriented_approach);

    flow_solver->run();
    const unsigned int n_adaptation_cycles = 1;
    
    for(unsigned int cycle = 0; cycle < n_adaptation_cycles; ++cycle)
    {
        anisotropic_mesh_adaptation->adapt_mesh();
        flow_solver->run();
        flow_solver->dg->output_results_vtk(1000 + cycle);
    }

    
    const auto mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::MappingQGeneric<dim, dim> mapping2(flow_solver->dg->high_order_grid->dof_handler_grid.get_fe().degree);
    dealii::hp::MappingCollection<dim> mapping_collection2(mapping2);
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients | dealii::update_jacobians | dealii::update_jacobian_pushed_forward_grads 
                                            | dealii::update_inverse_jacobians | dealii::update_jacobian_grads | dealii::update_quadrature_points | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, flow_solver->dg->fe_collection, flow_solver->dg->volume_quadrature_collection, update_flags);
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume2 (mapping_collection2, flow_solver->dg->fe_collection, flow_solver->dg->volume_quadrature_collection, dealii::update_hessians);
    
    PHiLiP::FEValuesShapeHessian<dim> fe_values_shape_hessian;
    for(const auto &cell : flow_solver->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        const unsigned int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        fe_values_collection_volume2.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        const dealii::FEValues<dim,dim> &fe_values_volume2 = fe_values_collection_volume2.get_present_fe_values();

        const unsigned int iquad = 0;
        std::cout<<"Dealii's FEValues shape_hessian = "<<fe_values_volume2.shape_hessian_component(0, iquad, 0)<<std::endl;
        
        fe_values_shape_hessian.reinit(fe_values_volume, iquad);
        std::cout<<"PHiLiP's FEValues shape_hessian = "<<fe_values_shape_hessian.shape_hessian_component(0, iquad, 0, fe_values_volume.get_fe())<<std::endl;

        std::cout<<std::endl;
    } // cell loop ends

    return 0;
}

//#if PHILIP_DIM==1
//template class AnisotropicMeshAdaptationCases <PHILIP_DIM,PHILIP_DIM>;
//#endif

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
