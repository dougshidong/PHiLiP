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
void AnisotropicMeshAdaptationCases<dim,nstate> :: verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const
{
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_jacobian_pushed_forward_grads | dealii::update_inverse_jacobians;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, update_flags);
    
    dealii::MappingQGeneric<dim, dim> mapping2(dg.high_order_grid->dof_handler_grid.get_fe().degree);
    dealii::hp::MappingCollection<dim> mapping_collection2(mapping2);
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume2 (mapping_collection2, dg.fe_collection, dg.volume_quadrature_collection, dealii::update_hessians);
    
    PHiLiP::FEValuesShapeHessian<dim> fe_values_shape_hessian;
    for(const auto &cell : dg.dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        const unsigned int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        fe_values_collection_volume2.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        const dealii::FEValues<dim,dim> &fe_values_volume2 = fe_values_collection_volume2.get_present_fe_values();
        
        const unsigned int n_dofs_cell = fe_values_volume.dofs_per_cell;
        const unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;
        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            fe_values_shape_hessian.reinit(fe_values_volume, iquad);
            
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
                dealii::Tensor<2,dim,double> shape_hessian_dealii = fe_values_volume2.shape_hessian_component(idof, iquad, istate);
                
                dealii::Tensor<2,dim,double> shape_hessian_philip = fe_values_shape_hessian.shape_hessian_component(idof, iquad, istate, fe_values_volume.get_fe());

                dealii::Tensor<2,dim,double> shape_hessian_diff = shape_hessian_dealii;
                shape_hessian_diff -= shape_hessian_philip;

                if(shape_hessian_diff.norm() > 1.0e-8)
                {
                    std::cout<<"Dealii's FEValues shape_hessian = "<<shape_hessian_dealii<<std::endl;
                    std::cout<<"PHiLiP's FEValues shape_hessian = "<<shape_hessian_philip<<std::endl;
                    std::cout<<"Frobenius norm of diff = "<<shape_hessian_diff.norm()<<std::endl;
                    std::cout<<"Aborting.."<<std::endl<<std::flush;
                    std::abort();
                }
            } // idof
        } // iquad
    } // cell loop ends

    pcout<<"PHiLiP's physical shape hessian matches that computed by dealii."<<std::endl;
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    const bool use_goal_oriented_approach = param.mesh_adaptation_param.use_goal_oriented_mesh_adaptation;
    const double complexity = param.mesh_adaptation_param.mesh_complexity_anisotropic_adaptation;
    const double normLp = param.mesh_adaptation_param.norm_Lp_anisotropic_adaptation;

    std::unique_ptr<AnisotropicMeshAdaptation<dim, nstate, double>> anisotropic_mesh_adaptation =
                        std::make_unique<AnisotropicMeshAdaptation<dim, nstate, double>> (flow_solver->dg, normLp, complexity, use_goal_oriented_approach);

    flow_solver->run();
    const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;
    
    for(unsigned int cycle = 0; cycle < n_adaptation_cycles; ++cycle)
    {
        anisotropic_mesh_adaptation->adapt_mesh();
        flow_solver->run();
    }

    verify_fe_values_shape_hessian(*(flow_solver->dg));

    const dealii::Point<dim> coordinates_of_highest_refined_cell = flow_solver->dg->coordinates_of_highest_refined_cell(false);

    pcout<<"Coordinates of highest refined cell = "<<coordinates_of_highest_refined_cell<<std::endl;

    dealii::Point<dim> expected_coordinates_of_highest_refined_cell;
    for(unsigned int i=0; i < dim; ++i) {
        expected_coordinates_of_highest_refined_cell[i] = 0.5;
    }
    const double distance_val  = expected_coordinates_of_highest_refined_cell.distance(coordinates_of_highest_refined_cell);
    pcout<<"Distance to the expected coordinates of the highest refined cell = "<<distance_val<<std::endl;

    int test_val = 0;
    if(distance_val > 0.1) {++test_val;}// should lie in a ball of radius 0.1
    return test_val;
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
    
