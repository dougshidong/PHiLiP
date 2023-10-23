#include <stdlib.h>
#include <iostream>
#include "anisotropic_mesh_adaptation_cases.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"
#include "mesh/mesh_adaptation/fe_values_shape_hessian.h"
#include "mesh/mesh_adaptation/mesh_error_estimate.h"
#include "mesh/mesh_adaptation/mesh_optimizer.hpp"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

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
double AnisotropicMeshAdaptationCases<dim,nstate> :: output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg) const
{
    dg->output_results_vtk(98765);

    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;

    return 0;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const double functional_exact = 0.0;
    std::shared_ptr< Functional<dim, nstate, double> > functional
                                = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
    const double functional_val = functional->evaluate_functional();
    const double error_val = abs(functional_val - functional_exact);
    return error_val;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    const bool run_mesh_optimizer = param.optimization_param.max_design_cycles > 0;
    const bool run_fixedfraction_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    
    flow_solver->run();
    flow_solver->use_polynomial_ramping = false;

    std::vector<double> functional_error_vector;
    std::vector<unsigned int> n_cycle_vector;
    std::vector<unsigned int> n_dofs_vector;

    const double functional_error_initial = evaluate_functional_error(flow_solver->dg);
    pcout<<"Functional error initial = "<<std::setprecision(16)<<functional_error_initial<<std::endl; // can be deleted later. 
    functional_error_vector.push_back(functional_error_initial);
    n_dofs_vector.push_back(flow_solver->dg->n_dofs());
    unsigned int current_cycle = 0;
    n_cycle_vector.push_back(current_cycle++);
    dealii::ConvergenceTable convergence_table;

    if(run_mesh_optimizer)
    {
        std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param, true);
        mesh_optimizer->run_full_space_optimizer();

        const double functional_error = evaluate_functional_error(flow_solver->dg);
        functional_error_vector.push_back(functional_error);
        n_dofs_vector.push_back(flow_solver->dg->n_dofs());
        n_cycle_vector.push_back(current_cycle++);

        convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
        convergence_table.add_value("functional_error",functional_error);
    }

    if(run_fixedfraction_mesh_adaptation)
    {
        const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;

        std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
        std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));

        for(unsigned int icycle = 0; icycle < n_adaptation_cycles; ++icycle)
        {
            meshadaptation->adapt_mesh();
            flow_solver->run();

            const double functional_error = evaluate_functional_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);

            convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table.add_value("functional_error",functional_error);
        }
    }

    output_vtk_files(flow_solver->dg);

    // output error vals
    pcout<<"\n cycles = [";
    for(long unsigned int i=0; i<n_cycle_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_cycle_vector[i];
    }
    pcout<<"];"<<std::endl;

    pcout<<"\n n_dofs = [";
    for(long unsigned int i=0; i<n_dofs_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_dofs_vector[i];
    }
    pcout<<"];"<<std::endl;

    std::string functional_type = "functional_error";
    pcout<<"\n "<<functional_type<<" = ["<<std::setprecision(16);
    for(long unsigned int i=0; i<functional_error_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<functional_error_vector[i];
    }
    pcout<<"];"<<std::endl;

    convergence_table.evaluate_convergence_rates("functional_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.set_scientific("functional_error", true);

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table.write_text(pcout.get_stream());}

return 0;
}

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
