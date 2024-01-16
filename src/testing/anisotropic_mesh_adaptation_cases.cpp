#include <stdlib.h>
#include <iostream>
#include "physics/euler.h"
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

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const
{
   const unsigned int degree_updated = 2;

    dg->high_order_grid->set_q_degree(degree_updated, true);
    dg->set_p_degree_and_interpolate_solution(degree_updated);
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::evaluate_regularization_matrix(
    dealii::TrilinosWrappers::SparseMatrix &regularization_matrix, 
    std::shared_ptr<DGBase<dim,double>> dg) const
{
    // Get volume of smallest element.
    const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[dg->high_order_grid->grid_degree];
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_vol(mapping, dg->high_order_grid->fe_metric_collection[dg->high_order_grid->grid_degree], volume_quadrature,
                    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_vol.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_values_vol.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_vol.dofs_per_cell);
    
    double min_cell_volume_local = 1.0e6;
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        double cell_vol = 0.0;
        fe_values_vol.reinit (cell);

        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }

        if(cell_vol < min_cell_volume_local)
        {
            min_cell_volume_local = cell_vol;
        }
    }

    const double min_cell_vol = dealii::Utilities::MPI::min(min_cell_volume_local, mpi_communicator);

    // Set sparsity pattern
    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dg->high_order_grid->dof_handler_grid,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    dealii::DynamicSparsityPattern dsp(dg->high_order_grid->dof_handler_grid.n_dofs(), dg->high_order_grid->dof_handler_grid.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dg->high_order_grid->dof_handler_grid, dsp, hanging_node_constraints);
    const dealii::IndexSet &locally_owned_dofs = dg->high_order_grid->locally_owned_dofs_grid;
    regularization_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, this->mpi_communicator);

    // Set elements.
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_vol.reinit (cell);
        cell->get_dof_indices(dofs_indices);
        cell_matrix = 0;
        
        double cell_vol = 0.0;
        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }
        const double omega_k = min_cell_vol/cell_vol;

        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int icomp = fe_values_vol.get_fe().system_to_component_index(i).first;
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int jcomp = fe_values_vol.get_fe().system_to_component_index(j).first;
                double val_ij = 0.0;

                if(icomp == jcomp)
                {
                    for(unsigned int q=0; q<n_quad_pts; ++q)
                    {
                        val_ij += omega_k*fe_values_vol.shape_grad(i,q)*fe_values_vol.shape_grad(j,q)*fe_values_vol.JxW(q);
                    }
                }
                cell_matrix(i,j) = val_ij;
            }
        }
        hanging_node_constraints.distribute_local_to_global(cell_matrix, dofs_indices, regularization_matrix); 
    } // cell loop ends
    regularization_matrix.compress(dealii::VectorOperation::add);
}

template <int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate> :: verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const
{
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_jacobian_pushed_forward_grads | dealii::update_inverse_jacobians;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, update_flags);
    
    dealii::MappingQGeneric<dim, dim> mapping2(dg.high_order_grid->get_current_fe_system().degree);
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
double AnisotropicMeshAdaptationCases<dim,nstate> :: output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int countval) const
{
    const int outputval = 7000 + countval;
    dg->output_results_vtk(outputval);

    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;

    return 0;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
/*
//    const double functional_exact = 0.1615892748498965;
    const double mach_inf = dg->all_parameters->euler_param.mach_inf;
    const double desnity_inf = 1.0;
    const double gam = 1.4;
    const double pressure_inf = 1.0/(gam * pow(mach_inf,2)); 
    const double tot_energy = pressure_inf/(gam - 1.0) + 0.5*desnity_inf;
    const double enthalpy_inf = (tot_energy + pressure_inf)/desnity_inf;

    const double domain_length = 1.4;
    double functional_exact = enthalpy_inf*domain_length;
*/
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
    dwr_error_val->total_dual_weighted_residual_error();
    return abs(dwr_error_val->net_functional_error);
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_enthalpy_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
if constexpr (nstate==dim+2)
{
    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                dg->all_parameters->euler_param.ref_length,
                dg->all_parameters->euler_param.gamma_gas,
                dg->all_parameters->euler_param.mach_inf,
                dg->all_parameters->euler_param.angle_of_attack,
                dg->all_parameters->euler_param.side_slip_angle);
    
    int overintegrate = 10;
    const unsigned int poly_degree = dg->get_min_fe_degree();
    dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    const unsigned int n_dofs_cell = fe_values_extra.dofs_per_cell;
    std::array<double,nstate> soln_at_q;

    double l2error = 0;
    double l1error = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    // Integrate solution error and output error
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution(dofs_indices[idof])*fe_values_extra.shape_value_component(idof,iquad,istate); 
            }
            
            const double pressure = euler_physics_double.compute_pressure(soln_at_q);
            const double enthalpy_at_q = euler_physics_double.compute_specific_enthalpy(soln_at_q,pressure);
            l2error += pow((enthalpy_at_q - euler_physics_double.enthalpy_inf),2) * fe_values_extra.JxW(iquad);
            l1error += pow(euler_physics_double.compute_entropy_measure(soln_at_q) - euler_physics_double.entropy_inf,2) * fe_values_extra.JxW(iquad);
        }
    } // cell loop ends
    const double l2error_global = sqrt(dealii::Utilities::MPI::sum(l2error, MPI_COMM_WORLD));
    const double l1error_global = sqrt(dealii::Utilities::MPI::sum(l1error, MPI_COMM_WORLD));
    (void) l2error_global;
    (void) l1error_global;
    return l2error_global;
}
std::abort();
return 0.0;
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    int output_val = 0;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    const bool run_mesh_optimizer = param.optimization_param.max_design_cycles > 0;
    const bool run_fixedfraction_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    
    flow_solver->run();
    output_vtk_files(flow_solver->dg, output_val++);
    //return 0;
    flow_solver->use_polynomial_ramping = false;

    std::vector<double> functional_error_vector;
    std::vector<double> enthalpy_error_vector;
    std::vector<unsigned int> n_cycle_vector;
    std::vector<unsigned int> n_dofs_vector;

    const double functional_error_initial = evaluate_functional_error(flow_solver->dg);
    //pcout<<"Functional error initial = "<<std::setprecision(16)<<functional_error_initial<<std::endl; // can be deleted later.
    const double enthalpy_error_initial = evaluate_enthalpy_error(flow_solver->dg);
    functional_error_vector.push_back(functional_error_initial);
    enthalpy_error_vector.push_back(enthalpy_error_initial);
    n_dofs_vector.push_back(flow_solver->dg->n_dofs());
    unsigned int current_cycle = 0;
    n_cycle_vector.push_back(current_cycle++);
    dealii::ConvergenceTable convergence_table_functional;
    dealii::ConvergenceTable convergence_table_enthalpy;
    if(run_mesh_optimizer)
    {
    // Run q1 optimizer.
        flow_solver->dg->freeze_artificial_dissipation=true;
        flow_solver->dg->set_p_degree_and_interpolate_solution(1);
        dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q1;
        evaluate_regularization_matrix(regularization_matrix_poisson_q1, flow_solver->dg);
   //     output_vtk_files(flow_solver->dg, output_val++);
        //for(unsigned int i=0; i<2; ++i)
        //{
            Parameters::AllParameters param_q1 = param;
            //param_q1.optimization_param.regularization_parameter = 5.0;
            //param_q1.optimization_param.regularization_scaling = 1.1;
            param_q1.optimization_param.max_design_cycles = 3;
            
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q1 = 
                            std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param_q1, true);
            mesh_optimizer_q1->run_full_space_optimizer(regularization_matrix_poisson_q1);
            flow_solver->run();
            
            increase_grid_degree_and_interpolate_solution(flow_solver->dg);
            dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2;
            evaluate_regularization_matrix(regularization_matrix_poisson_q2, flow_solver->dg);
            
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q2 = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param, true);
            mesh_optimizer_q2->run_full_space_optimizer(regularization_matrix_poisson_q2);
            

            const double functional_error = evaluate_functional_error(flow_solver->dg);
            const double enthalpy_error = evaluate_enthalpy_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            enthalpy_error_vector.push_back(enthalpy_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);

            convergence_table_functional.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_functional.add_value("functional_error",functional_error);
            convergence_table_enthalpy.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_enthalpy.add_value("enthalpy_error",enthalpy_error);

     //       increase_grid_degree_and_interpolate_solution(flow_solver->dg);
     //       output_vtk_files(flow_solver->dg, output_val++);
            /*      
            auto mesh_adaptation_param2 = param.mesh_adaptation_param;
            mesh_adaptation_param2.use_goal_oriented_mesh_adaptation = false;
            mesh_adaptation_param2.refine_fraction = 1.0;
            std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
            std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(mesh_adaptation_param2));
            meshadaptation->adapt_mesh();
            flow_solver->run();
            */
        //}
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
            const double enthalpy_error = evaluate_enthalpy_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            enthalpy_error_vector.push_back(enthalpy_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);

            convergence_table_functional.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_functional.add_value("functional_error",functional_error);
            convergence_table_enthalpy.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table_enthalpy.add_value("enthalpy_error",enthalpy_error);
        }
    }

    output_vtk_files(flow_solver->dg, output_val++);

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
    
    std::string errortype = "enthalpy_error";
    pcout<<"\n "<<errortype<<" = ["<<std::setprecision(16);
    for(long unsigned int i=0; i<enthalpy_error_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<enthalpy_error_vector[i];
    }
    pcout<<"];"<<std::endl;

    convergence_table_functional.evaluate_convergence_rates("functional_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table_functional.set_scientific("functional_error", true);
    convergence_table_enthalpy.evaluate_convergence_rates("enthalpy_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table_enthalpy.set_scientific("enthalpy_error", true);

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary for functional error" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table_functional.write_text(pcout.get_stream());}
    
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary for enthalpy error" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table_enthalpy.write_text(pcout.get_stream());}

return 0;
}

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
