#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <numeric>  
#include <cmath>    

// --- PHiLiP & DEAL.II HEADERS ---
#include "lpcvt_mesh_adaptation.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"
#include "mesh/mesh_adaptation/fe_values_shape_hessian.h"
#include "mesh/mesh_adaptation/mesh_error_estimate.h"
#include "mesh/mesh_adaptation/mesh_optimizer.hpp"
#include "mesh/mesh_adaptation/meshes_interpolation.h"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include "grid_refinement/reconstruct_poly.h"
#include "grid_refinement/field.h"
#include "grid_refinement/grid_refinement_continuous.h"
#include "grid_refinement/grid_refinement.h"
#include "functional/adjoint.h"
#include "physics/physics_factory.h"
#include "physics/model_factory.h"
#include "ADTypes.hpp"
#include "grid_refinement/size_field.h"
#include "grid_refinement/msh_out.h"
#include "ode_solver/ode_solver_factory.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_generator.h>
#include "post_processor/physics_post_processor.h"
#include "functional/functional.h"
#include "functional/adjoint.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {
    namespace Tests {

        /**
         * Constructor for LpCVT mesh adaptation test cases
         * Initializes the test with given parameters and parameter handler
         */
        template <int dim, int nstate>
        LpCVTMeshAdaptationCases<dim, nstate>::LpCVTMeshAdaptationCases(
            const Parameters::AllParameters* const parameters_input,
            const dealii::ParameterHandler& parameter_handler_input)
            : TestsBase::TestsBase(parameters_input)
            , parameter_handler(parameter_handler_input)
        {}

        /**
         * Evaluates the absolute dual-weighted residual (DWR) error
         * This provides an error estimate that combines primal and dual solutions
         * Used for goal-oriented error estimation
         */
        template <int dim, int nstate>
        double LpCVTMeshAdaptationCases<dim, nstate>::evaluate_abs_dwr_error(
            std::shared_ptr<DGBase<dim, double>> dg) const
        {
            // Create DWR error estimator
            std::unique_ptr<DualWeightedResidualError<dim, nstate, double>> dwr_error_val =
                std::make_unique<DualWeightedResidualError<dim, nstate, double>>(dg);
            
            // Compute and return total DWR error
            const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
            return abs_dwr_error;
        }

        template <int dim, int nstate>
        std::array<double,nstate> LpCVTMeshAdaptationCases<dim, nstate> :: evaluate_soln_exact(const dealii::Point<dim> &point) const
        {
            std::array<double, nstate> soln_exact;
            const double x = point[0];
            const double y = point[1];
            const double a = -0.4;
            const double b = 0.4;
            const double c = 3.0/8.0;
            const double x_on_curve = a*pow(y,2) + b*y + c;

            // Get u0 exact
            soln_exact[0] = 0.0;
            if(x <= x_on_curve)
            {
                soln_exact[0] = 1.0;
            }

            // Get u1 exact
            const bool region_1 = (y > (1.0-x)) && (x <=x_on_curve);
            const bool region_2 = (y <= (1.0-x)) && (x <= x_on_curve);
            const bool region_3 = (y <= (1.0-x)) && (x > x_on_curve);
            const bool region_4 = (x > x_on_curve) && (y < (11.0/8.0 - x)) && (y > (1.0-x));
            const bool region_5 = y >= (11.0/8.0 - x);
            const double y_tilde = (-(b+1.0) + sqrt(pow(b+1.0,2) - 4.0*a*(c-x-y)))/(2.0*a);
            const double x_tilde = a*pow(y_tilde,2) + b*y_tilde + c;
            const double u1_bc = 0.3;
            if(region_1)
            {
                soln_exact[1] = u1_bc + (1.0-y);
            }
            else if(region_2)
            {
                soln_exact[1] = u1_bc + x;
            }
            else if(region_3)
            {
                soln_exact[1] = u1_bc + x_tilde;
            }
            else if(region_4)
            {
                soln_exact[1] = u1_bc + (1.0 - y_tilde);
            }
            else if(region_5)
            {
                soln_exact[1] = u1_bc;
            }
            else
            {
                std::cout<<"The domain is completely covered by regions 1 to 5. Shouldn't have reached here. Aborting.."<<std::endl;
                std::abort();
            }
            return soln_exact;
        }

        // /**
        //  * Evaluates the solution error in L2 norm
        //  * Compares numerical solution to known exact solution
        //  * Used to quantify accuracy of the computed solution
        //  */
        // template <int dim, int nstate>
        // std::array<double, nstate> LpCVTMeshAdaptationCases<dim, nstate>::evaluate_L2_error_norm(
        //     std::shared_ptr<DGBase<dim, double>> dg) const
        // {
        //     // Initialize error array
        //     std::array<double, nstate> l2_error_per_state;
        //     std::fill(l2_error_per_state.begin(), l2_error_per_state.end(), 0.0);
            
        //     // Create physics with manufactured solution
        //     std::shared_ptr<Physics::ModelBase<dim, nstate, double>> model = 
        //         Physics::ModelFactory<dim, nstate, double>::create_Model(all_parameters);
        //     std::shared_ptr<Physics::PhysicsBase<dim, nstate, double>> physics_double = 
        //         Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters, model);
            
        //     if (!physics_double->manufactured_solution_function) {
        //         pcout << "WARNING: Manufactured solution function not available" << std::endl;
        //         return l2_error_per_state;
        //     }
            
        //     // Overintegration parameter for accurate error computation
        //     const int overintegrate = 10;
            
        //     // Initialize per-state L2 norms
        //     std::array<double, nstate> l2_norm_squared;
        //     std::fill(l2_norm_squared.begin(), l2_norm_squared.end(), 0.0);
            
        //     // Loop over all locally owned cells
        //     for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell) {
        //         if (!cell->is_locally_owned()) continue;
                
        //         // Get polynomial degree of current cell
        //         const unsigned int poly_degree = dg->fe_collection[cell->active_fe_index()].degree;
                
        //         // Create quadrature rule with overintegration
        //         dealii::QGauss<dim> quad_extra(poly_degree + overintegrate);
                
        //         // Create FEValues object for solution evaluation
        //         dealii::FEValues<dim, dim> fe_values_extra(
        //             *(dg->high_order_grid->mapping_fe_field), 
        //             dg->fe_collection[cell->active_fe_index()], 
        //             quad_extra,
        //             dealii::update_values | 
        //             dealii::update_JxW_values | 
        //             dealii::update_quadrature_points);
                
        //         // Reinitialize for current cell
        //         fe_values_extra.reinit(cell);
                
        //         // Get DOF indices for current cell
        //         std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
        //         cell->get_dof_indices(dofs_indices);
                
        //         const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
                
        //         // Loop over quadrature points
        //         for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
        //             // Evaluate numerical solution at quadrature point
        //             std::array<double, nstate> soln_at_q;
        //             std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    
        //             for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
        //                 const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
        //                 soln_at_q[istate] += dg->solution[dofs_indices[idof]] * 
        //                                     fe_values_extra.shape_value_component(idof, iquad, istate);
        //             }
                    
        //             // Get physical quadrature point
        //             const dealii::Point<dim> qpoint = fe_values_extra.quadrature_point(iquad);
                    
        //             // Compute error for each state component
        //             for (unsigned int istate = 0; istate < nstate; ++istate) {
        //                 // Evaluate exact solution
        //                 const double uexact = physics_double->manufactured_solution_function->value(qpoint, istate);
                        
        //                 // Accumulate L2 error squared
        //                 const double error = soln_at_q[istate] - uexact;
        //                 l2_norm_squared[istate] += error * error * fe_values_extra.JxW(iquad);
        //             }
        //         }
        //     }
            
        //     // Sum across all MPI processes and take square root
        //     for (unsigned int istate = 0; istate < nstate; ++istate) {
        //         const double global_l2_squared = dealii::Utilities::MPI::sum(l2_norm_squared[istate], mpi_communicator);
        //         l2_error_per_state[istate] = std::sqrt(global_l2_squared);
        //     }
            
        //     // Output results
        //     pcout << "L2 Error Norms:" << std::endl;
        //     for (unsigned int istate = 0; istate < nstate; ++istate) {
        //         pcout << "  State " << istate << ": " << l2_error_per_state[istate] << std::endl;
        //     }
            
        //     // Also compute and report total L2 error (all states combined)
        //     double total_l2_squared = 0.0;
        //     for (unsigned int istate = 0; istate < nstate; ++istate) {
        //         total_l2_squared += l2_error_per_state[istate] * l2_error_per_state[istate];
        //     }
        //     const double total_l2_error = std::sqrt(total_l2_squared);
        //     pcout << "  Total L2 error: " << total_l2_error << std::endl;
            
        //     return l2_error_per_state;
        // }


        /**
         * Evaluates the functional error compared to known exact value
         * This is specific to the test case and used to track convergence
         */
        template <int dim, int nstate>
        double LpCVTMeshAdaptationCases<dim, nstate>::evaluate_functional_error(
            std::shared_ptr<DGBase<dim,double>> dg) const
        {
            // // Known exact functional value for the test case
            // const double functional_exact = 0.1512447195285363; // with heaviside
            // //const double functional_exact = 0.1792480962990282; // without heaviside
            
            // // Create functional based on parameters
            // std::shared_ptr< Functional<dim, nstate, double> > functional
            //     = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
            
            // // Evaluate current functional value
            // const double functional_val = functional->evaluate_functional();
            // std::cout<<"Functional from surface integral = "<<std::setprecision(16)<<functional_exact<<std::endl;
            // std::cout<<"Functional from functional class = "<<std::setprecision(16)<<functional_val<<std::endl;
            // // Return absolute error
            // const double error_val = abs(functional_exact - functional_val);
            // return error_val;

            int overintegrate = 500;
            const unsigned int poly_degree = dg->get_min_fe_degree();
            dealii::QGauss<dim-1> face_quad_extra(overintegrate);
            const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
            dealii::FEFaceValues<dim,dim> fe_face_values_extra(mapping, dg->fe_collection[poly_degree], face_quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_face_quad_pts = fe_face_values_extra.n_quadrature_points;
            
            double functional_local = 0.0;

            // Integrate solution error and output error
            for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
            {
                if (!cell->is_locally_owned()) continue;

                for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
                {
                    auto face = cell->face(iface);
                    if(face->at_boundary())
                    {
                        const unsigned int boundary_id = face->boundary_id();
                        if(boundary_id == 1)
                        {
                            fe_face_values_extra.reinit (cell, iface);

                            for(unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad)
                            {
                                const dealii::Point<dim> &phys_point = fe_face_values_extra.quadrature_point(iquad);
                                const std::array<double,nstate> soln_exact_at_q = evaluate_soln_exact(phys_point);
                                if( abs(phys_point[0] - 1.0) > 1.0e-15)
                                {
                                    std::cout<<"Not at right boundary. Aborting..."<<std::endl;
                                    std::cout<<"x = "<<std::setprecision(16)<<phys_point[0]<<std::endl;
                                    std::cout<<"error in x = "<<std::setprecision(16)<<abs(1.0-phys_point[0])<<std::endl;
                                    std::abort();
                                }
                                
                                //=======================================================================
                                // Evaluate continuous logistic heaviside.
                                const double y = phys_point[1];
                                const double yc = 0.05;
                                const double heaviside_min = 1.0e-5; // heaviside at y=0
                                const double logterm = log(1/heaviside_min - 1.0);
                                const double epsilon_val = yc/logterm;
                                const double heaviside_at_y = 1.0/(1.0 + exp(-(y-yc)/epsilon_val));
                                //=======================================================================
                                const double integrand = heaviside_at_y * pow(soln_exact_at_q[1],2);
                                functional_local += integrand*fe_face_values_extra.JxW(iquad);                        
                            } // iquad ends
                            
                        } // if (boundary_id==1) ends
                    } // if (face->at_boundary()) ends
                } // face loop ends
            } // cell loop ends

            // Create functional based on parameters
            std::shared_ptr< Functional<dim, nstate, double> > functional
                = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
            
            // Evaluate current functional value
            const double functional_val = functional->evaluate_functional();

            const double functional_global = dealii::Utilities::MPI::sum(functional_local, MPI_COMM_WORLD);
            std::cout<<"Functional from surface integral = "<<std::setprecision(16)<<functional_global<<std::endl;
            std::cout<<"Functional from functional class = "<<std::setprecision(16)<<functional_val<<std::endl;
            // Return absolute error
            const double error_val = abs(functional_global - functional_val);
            return error_val;

        }

        template<int dim, int nstate>
        void LpCVTMeshAdaptationCases<dim,nstate>::evaluate_regularization_matrix(
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
        /**
         * Main inner adaptation loop
         * Runs mesh optimization cycles until convergence or max cycles reached
         * Returns false if LpCVT reconstruction is needed (optimization stalls)
         */
        template <int dim, int nstate>
        bool LpCVTMeshAdaptationCases<dim, nstate>::run_adaptation_loop(
            std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
            const Parameters::AllParameters& param,
            const unsigned int max_cycles,
            std::vector<double>& functional_error_vector,
            std::vector<unsigned int> n_dofs_vector,
            std::vector<unsigned int>& n_cycle_vector,
            const unsigned int outer_iter
            ) const
        {
            dealii::ConvergenceTable convergence_table;
            const bool run_mesh_optimizer = true;  // Flag to enable mesh optimization

            // Open file for writing convergence data
            std::ofstream convergence_file;
            if (mpi_rank == 0) {
                std::string filename = "convergence_data_outer_" + std::to_string(outer_iter) + ".txt";
                convergence_file.open(filename);
                convergence_file << std::scientific << std::setprecision(16);
                convergence_file << "# Cycle DOFs Functional_Error\n";
            }

            // Run adaptation cycles (inner loop)
            for (unsigned int cycle = 0; cycle < max_cycles; ++cycle)
            {
                pcout << "\n--- Adaptation cycle " << cycle + 1 << "/" << max_cycles << " ---" << std::endl;
                
                //Pre-optimization evaluation
                const double pre_functional_error = evaluate_functional_error(flow_solver->dg);
                const unsigned int pre_dofs = flow_solver->dg->n_dofs();
                pcout << "Pre-adaptation functional error: " << pre_functional_error << std::endl;
                
                functional_error_vector.push_back(pre_functional_error);
                n_dofs_vector.push_back(pre_dofs);
                n_cycle_vector.push_back(outer_iter + 0.5);
                
                // Write pre-optimization data
                if (mpi_rank == 0) {
                    convergence_file << (outer_iter + 0.5) << " " << pre_dofs << " " << pre_functional_error << "\n";
                    convergence_file.flush();
                }

                if (run_mesh_optimizer)
                {
                    pcout << "Running full-space mesh optimizer..." << std::endl;
                    //double mesh_weight = param.optimization_param.mesh_weight_factor;
                    
                    try {
                        // Create and run mesh optimizer
                        dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q1;
                        evaluate_regularization_matrix(regularization_matrix_poisson_q1, flow_solver->dg);
                        Parameters::AllParameters param_q1 = param;
                        param_q1.optimization_param.max_design_cycles = 1;
                        param_q1.optimization_param.regularization_parameter_sim = 1.0;
                        param_q1.optimization_param.regularization_parameter_control = 1.0;
                        const bool use_oneD_parameteriation = false;
                        const bool output_refined_nodes = false;
                        std::unique_ptr<MeshOptimizer<dim, nstate>> mesh_optimizer =
                            std::make_unique<MeshOptimizer<dim, nstate>>(flow_solver->dg, &param_q1, true);
                        // write_dwr_cellwise_vtk(flow_solver);
                        mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson_q1, use_oneD_parameteriation, output_refined_nodes,0);
                        //mesh_weight = 0.0;  // Reset weight after first iteration
                        pcout << "Mesh optimizer completed successfully." << std::endl;
                        flow_solver->run();

                    }
                    catch (const std::exception& e) {
                        pcout << "WARNING: Mesh optimizer failed with error: " << e.what() << std::endl;
                        flow_solver->run();
                        write_dwr_cellwise_vtk(flow_solver);
                        const double functional_error = evaluate_functional_error(flow_solver->dg);
                        const unsigned int post_dofs = flow_solver->dg->n_dofs();
                        
                        functional_error_vector.push_back(functional_error);
                        n_dofs_vector.push_back(post_dofs);
                        n_cycle_vector.push_back(outer_iter + 1);
                        
                        pcout<<"Current cycle = "<<(outer_iter + 1)<<";  Functional error = "<<functional_error<<std::endl;
                        
                        // Write post-optimization data
                        if (mpi_rank == 0) {
                            convergence_file << (outer_iter + 1) << " " << post_dofs << " " << functional_error << "\n";
                            convergence_file.flush();
                        }
                        
                        //Add to convergence table for analysis
                        convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
                        convergence_table.add_value("functional_error",functional_error);

                        std::string error_msg = e.what();
                        // Check if optimization stalled
                        if (error_msg.find("step size too small") != std::string::npos) {
                            pcout << "Optimization stopped due to small step size. Need LpCVT reconstruction." << std::endl;
                            if (mpi_rank == 0) convergence_file.close();
                            return false;
                        }

                        // For any other error, stop the adaptation
                        pcout << "Other error encountered, stopping adaptation." << std::endl;
                        if (mpi_rank == 0) convergence_file.close();
                        return false;
                    }
                }
                else
                {
                    // Alternative: just run flow solver without optimization
                    pcout << "Running flow solver..." << std::endl;
                    flow_solver->run();
                }
            }
            const double functional_error = evaluate_functional_error(flow_solver->dg);
            const unsigned int post_dofs = flow_solver->dg->n_dofs();
            
            functional_error_vector.push_back(functional_error);
            n_dofs_vector.push_back(post_dofs);
            n_cycle_vector.push_back(outer_iter + 1);
            
            pcout<<"Current cycle = "<<(outer_iter + 1)<<";  Functional error = "<<functional_error<<std::endl;
            
            // Write post-optimization data
            if (mpi_rank == 0) {
                convergence_file << (outer_iter + 1) << " " << post_dofs << " " << functional_error << "\n";
                convergence_file.flush();
            }
            
            // Add to convergence table for analysis
            convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table.add_value("functional_error",functional_error);
            
            return true;  // Adaptation completed successfully
        }




        /**
         * Extracts the metric field from the current solution
         * Uses GridRefinement infrastructure to compute anisotropic metric tensor
         * This metric guides the LpCVT mesh generation
         */
        template <int dim, int nstate>
        typename LpCVTMeshAdaptationCases<dim, nstate>::MetricData
            LpCVTMeshAdaptationCases<dim, nstate>::extract_metric_field(
                const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
                const unsigned int outer_loop,
                const Parameters::AllParameters& param) const
        {
            pcout << "\n--- Extracting Metric Field Using GridRefinement ---" << std::endl;

            MetricData metric_data;

            try {
                auto dg = flow_solver->dg;

                // Update high-order grid mapping
                dg->high_order_grid->update_mapping_fe_field();
                dg->high_order_grid->volume_nodes.update_ghost_values();

                // Use the parsed grid refinement parameters
                Parameters::GridRefinementParam gr_param = param.grid_refinement_study_param.grid_refinement_param_vector[0];
                
                // Override error indicator based on goal-oriented adaptation setting
                if (param.mesh_adaptation_param.use_goal_oriented_mesh_adaptation) {
                    gr_param.error_indicator = Parameters::GridRefinementParam::ErrorIndicator::adjoint_based;
                }

                // Handle complexity for this iteration
                double target_complexity;
                
                // Option 1: If complexity_vector has values for all iterations, use them directly
                if (outer_loop < gr_param.complexity_vector.size()) {
                    target_complexity = gr_param.complexity_vector[outer_loop];
                }
                // Option 2: Use base complexity with scaling factor
                else {
                    double base_complexity = gr_param.complexity_vector.empty() ? 
                        2000.0 : gr_param.complexity_vector[0];
                    // Use complexity_scale as growth factor
                    target_complexity = base_complexity * std::pow(gr_param.complexity_scale, outer_loop);
                }
                
                // Update complexity for this iteration
                gr_param.complexity_vector.clear();
                gr_param.complexity_vector.push_back(target_complexity);
                
                pcout << "    Target complexity for outer loop " << outer_loop 
                    << ": " << target_complexity << std::endl;
                
                // These parameters should already be set from the file
                pcout << "Grid refinement parameters from file:" << std::endl;
                pcout << "  Refinement method: " 
                    << (gr_param.refinement_method == Parameters::GridRefinementParam::RefinementMethod::continuous ? 
                        "continuous" : "other") << std::endl;
                pcout << "  Error indicator: " 
                    << (gr_param.error_indicator == Parameters::GridRefinementParam::ErrorIndicator::adjoint_based ?
                        "adjoint_based" : "hessian_based") << std::endl;
                pcout << "  Anisotropic: " << (gr_param.anisotropic ? "true" : "false") << std::endl;
                pcout << "  Anisotropic ratio range: [" << gr_param.anisotropic_ratio_min
                    << ", " << gr_param.anisotropic_ratio_max << "]" << std::endl;
                pcout << "  Norm Lq: " << gr_param.norm_Lq << std::endl;
                pcout << "  r_max: " << gr_param.r_max << ", c_max: " << gr_param.c_max << std::endl;

                // Create physics objects needed for error estimation
                std::shared_ptr<Physics::ModelBase<dim, nstate, double>> model =
                    Physics::ModelFactory<dim, nstate, double>::create_Model(&param);
                std::shared_ptr<Physics::PhysicsBase<dim, nstate, double>> physics =
                    Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&param, model);

                // For adjoint-based refinement, create functional and adjoint objects
                std::shared_ptr<Functional<dim, nstate, double>> functional = nullptr;
                std::shared_ptr<Adjoint<dim, nstate, double>> adjoint = nullptr;

                if (gr_param.error_indicator == Parameters::GridRefinementParam::ErrorIndicator::adjoint_based) {
                    // Goal-oriented adaptation path
                    // Create functional based on parameters
                    functional = FunctionalFactory<dim, nstate, double>::create_Functional(
                        param.functional_param, dg);

                    // Create adjoint solver with automatic differentiation types
                    using ADtype = Sacado::Fad::DFad<double>;
                    std::shared_ptr<Physics::ModelBase<dim, nstate, ADtype>> model_adtype =
                        Physics::ModelFactory<dim, nstate, ADtype>::create_Model(&param);
                    std::shared_ptr<Physics::PhysicsBase<dim, nstate, ADtype>> physics_adtype =
                        Physics::PhysicsFactory<dim, nstate, ADtype>::create_Physics(&param, model_adtype);

                    adjoint = std::make_shared<Adjoint<dim, nstate, double>>(dg, functional, physics_adtype);

                    // Create GridRefinement_Continuous with adjoint for goal-oriented metric
                    auto grid_refinement = std::make_unique<GridRefinement::GridRefinement_Continuous<dim, nstate, double>>(
                        gr_param, adjoint, physics);

                    // Compute the metric field based on adjoint-weighted error
                    grid_refinement->field();

                    // Output VTK for visualization (note: uses default subdivision)
                    pcout << "Writing detailed VTK output..." << std::endl;
                    grid_refinement->output_results_vtk(outer_loop);
                    pcout << "VTK output written successfully." << std::endl;

                    // Extract the computed h_field (contains metric tensor)
                    metric_data.h_field = grid_refinement->get_h_field();
                }
                else {
                    // Solution-oriented adaptation path (Hessian-based)
                    // Create GridRefinement_Continuous without adjoint
                    auto grid_refinement = std::make_unique<GridRefinement::GridRefinement_Continuous<dim, nstate, double>>(
                        gr_param, dg, physics);

                    // Compute the metric field based on solution Hessian
                    grid_refinement->field();

                    // Output VTK for visualization
                    pcout << "Writing detailed VTK output..." << std::endl;
                    grid_refinement->output_results_vtk(outer_loop);
                    pcout << "VTK output written successfully." << std::endl;

                    // Extract the computed h_field
                    metric_data.h_field = grid_refinement->get_h_field();
                }

                // Validate extraction
                metric_data.valid = (metric_data.h_field != nullptr);

                if (metric_data.valid) {
                    pcout << "--- Metric Field Extraction Complete ---" << std::endl;

                    // Print statistics about the metric field
                    auto scale_vec = metric_data.h_field->get_scale_vector();
                    pcout << "  Number of cells: " << metric_data.h_field->size() << std::endl;
                    pcout << "  Scale range: [" << *std::min_element(scale_vec.begin(), scale_vec.end())
                        << ", " << *std::max_element(scale_vec.begin(), scale_vec.end()) << "]" << std::endl;

                    // If anisotropic, print aspect ratio information
                    if (gr_param.anisotropic) {
                        auto aniso_ratios = metric_data.h_field->get_max_anisotropic_ratio_vector_dealii();
                        double min_ratio = *std::min_element(aniso_ratios.begin(), aniso_ratios.end());
                        double max_ratio = *std::max_element(aniso_ratios.begin(), aniso_ratios.end());
                        pcout << "  Anisotropic ratio range: [" << min_ratio << ", " << max_ratio << "]" << std::endl;
                    }
                }
                else {
                    pcout << "ERROR: Failed to extract h_field from GridRefinement" << std::endl;
                }
            }
            catch (const std::exception& e) {
                pcout << "ERROR extracting metric field: " << e.what() << std::endl;
                metric_data.valid = false;
            }

            return metric_data;
        }

        /**
         * Writes cell-wise DWR error estimates to VTK format
         * Used for visualization and debugging of error distribution
         */
        template <int dim, int nstate>
        void LpCVTMeshAdaptationCases<dim, nstate>::write_dwr_cellwise_vtk(
            const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver) const
        {
            try {
                // Create DWR error estimator
                // The constructor automatically computes and writes VTK files
                auto dwr = std::make_unique<DualWeightedResidualError<dim, nstate, double>>(flow_solver->dg);

                // Compute DWR (this also triggers VTK output internally)
                (void)dwr->total_dual_weighted_residual_error();

                pcout << "DWR-fine VTKs written by DualWeightedResidualError." << std::endl;
            } catch (const std::exception& e) {
                pcout << "WARNING: DWR write failed: " << e.what() << std::endl;
            }
        }


        // template <int dim, int nstate>
        // void LpCVTMeshAdaptationCases<dim, nstate>::write_lpcvt_background_mesh(
        //     const std::string& filename,
        //     const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
        //     const MetricData& metric_data) const
        // {
        //     pcout << "Writing LpCVT-compatible background mesh to: " << filename << std::endl;

        //     std::ofstream out_msh(filename);
        //     out_msh << std::scientific << std::setprecision(16);

        //     PHiLiP::GridRefinement::MshOut<dim, double> msh_out(flow_solver->dg->dof_handler);


        //     if (metric_data.valid && metric_data.h_field)
        //     {
        //         // 1. Get the original metric vector. Its order is based on active_cell_index().
        //         const std::vector<dealii::Tensor<2, dim>>& original_metrics =
        //             metric_data.h_field->get_inverse_metric_vector();

        //         // 2. Create a new vector to hold the metrics in the correct iteration order.
        //         std::vector<dealii::Tensor<2, dim>> reordered_metrics;
        //         reordered_metrics.reserve(flow_solver->dg->triangulation->n_active_cells());

        //         // 3. Iterate through the cells in the same order that MshOut will use for writing geometry.
        //         for (const auto& cell : flow_solver->dg->dof_handler.active_cell_iterators())
        //         {
        //             // For each cell in the iteration, get its unique index.
        //             const unsigned int cell_index = cell->active_cell_index();

        //             // Use the cell's unique index to look up the correct metric from the original
        //             // data and add it to the new, correctly-ordered vector.
        //             reordered_metrics.push_back(original_metrics[cell_index]);
        //         }

        //         // 4. Add the REORDERED metric vector to the writer.
        //         msh_out.add_data_vector(reordered_metrics,
        //             PHiLiP::GridRefinement::StorageType::element,
        //             "metric");

        //     }

        //     // Make sure mapping reflects the current high-order geometry
        //     flow_solver->dg->high_order_grid->update_mapping_fe_field();

        //     auto& tria = *flow_solver->dg->triangulation;
        //     auto& mapping = *flow_solver->dg->high_order_grid->mapping_fe_field;

        //     auto& vertices = const_cast<std::vector<dealii::Point<dim>>&>(tria.get_vertices());
        //     for (auto cell : flow_solver->dg->dof_handler.active_cell_iterators()) {
        //         for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
        //             const unsigned int vidx = cell->vertex_index(v);
        //             const dealii::Point<dim> unit = dealii::GeometryInfo<dim>::unit_cell_vertex(v);
        //             const dealii::Point<dim> real = mapping.transform_unit_to_real_cell(cell, unit);
        //             vertices[vidx] = real;
        //         }
        //     }

        //     msh_out.write_msh(out_msh);

        //     pcout << "Mesh and metric field written successfully." << std::endl;
        // }

        
        /**
         * Comparison functor for dealii::Point objects
         * Used for creating ordered maps of vertices
         */
        template <int dim>
        struct PointLess {
            bool operator()(const dealii::Point<dim>& a, const dealii::Point<dim>& b) const {
                // Lexicographic ordering: compare dimension by dimension
                for (int d = 0; d < dim; ++d) {
                    if (a[d] < b[d]) return true;
                    if (a[d] > b[d]) return false;
                }
                return false;
            }
        };

        /**
         * Writes background mesh with metric field for LpCVT solver
         * This creates a serial mesh file that the external LpCVT Python code can read
         * Includes MPI gathering to collect distributed mesh on rank 0
         */
        template <int dim, int nstate>
        void LpCVTMeshAdaptationCases<dim, nstate>::write_lpcvt_background_mesh(
            const std::string& filename,
            const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
            const MetricData& metric_data) const
        {
            pcout << "Writing LpCVT-compatible background mesh to: " << filename << std::endl;
            
            const auto& tria = *flow_solver->dg->triangulation;
            const unsigned int verts_per_cell = dealii::GeometryInfo<dim>::vertices_per_cell;
            const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
            const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

            // Update high-order geometry to get accurate vertex positions
            flow_solver->dg->high_order_grid->update_mapping_fe_field();
            flow_solver->dg->high_order_grid->volume_nodes.update_ghost_values();
            
            // Update vertex positions based on high-order mapping
            auto& vertices = const_cast<std::vector<dealii::Point<dim>>&>(tria.get_vertices());
            auto &mapping = *flow_solver->dg->high_order_grid->mapping_fe_field;
            
            // Transform vertices to physical space for curved elements
            for (auto cell = flow_solver->dg->dof_handler.begin_active();
                cell != flow_solver->dg->dof_handler.end(); ++cell)
            {
                if (!(cell->is_locally_owned() || cell->is_ghost()))
                    continue;

                for (unsigned int v = 0; v < verts_per_cell; ++v)
                {
                    const unsigned int vidx = cell->vertex_index(v);
                    const dealii::Point<dim> unit = dealii::GeometryInfo<dim>::unit_cell_vertex(v);
                    const dealii::Point<dim> real = mapping.transform_unit_to_real_cell(cell, unit);
                    vertices[vidx] = real;
                }
            }

            // --- MPI GATHERING PHASE 1: VERTICES WITH DEDUPLICATION ---
            
            // Collect indices of vertices used by locally owned cells
            std::set<unsigned int> local_vertex_indices;
            for (auto cell = tria.begin_active(); cell != tria.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
                for (unsigned int v = 0; v < verts_per_cell; ++v) {
                    local_vertex_indices.insert(cell->vertex_index(v));
                }
            }
            
            // Build list of vertices actually used and create mapping
            std::vector<dealii::Point<dim>> local_used_vertices;
            std::map<unsigned int, unsigned int> old_to_local_map;
            for (unsigned int vidx : local_vertex_indices) {
                old_to_local_map[vidx] = local_used_vertices.size();
                local_used_vertices.push_back(vertices[vidx]);
            }
            
            // Flatten vertex coordinates for MPI
            const unsigned int n_local = local_used_vertices.size();
            std::vector<double> local_coords(n_local * dim);
            for (unsigned int i = 0; i < n_local; ++i) {
                for (int d = 0; d < dim; ++d) {
                    local_coords[i * dim + d] = local_used_vertices[i][d];
                }
            }
            
            // Gather vertex counts
            std::vector<int> sizes(n_mpi);
            int local_size = static_cast<int>(n_local);
            MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, mpi_communicator);
            
            // Prepare for gathering coordinates
            std::vector<double> all_coords;
            std::vector<dealii::Point<dim>> unique_vertices;
            std::vector<unsigned int> local_to_global;
            
            if (mpi_rank == 0) {
                std::vector<int> recvcounts(n_mpi), displs(n_mpi, 0);
                for (int r = 0; r < n_mpi; ++r) {
                    recvcounts[r] = sizes[r] * dim;
                }
                for (int r = 1; r < n_mpi; ++r) {
                    displs[r] = displs[r - 1] + recvcounts[r - 1];
                }
                int total_coord_size = (n_mpi > 0) ? (displs.back() + recvcounts.back()) : 0;
                all_coords.resize(total_coord_size);
                
                // Gather all coordinates
                MPI_Gatherv(local_coords.data(), local_coords.size(), MPI_DOUBLE,
                            all_coords.data(), recvcounts.data(), displs.data(), 
                            MPI_DOUBLE, 0, mpi_communicator);
                
                // Deduplicate vertices with tolerance
                const double tolerance = 1e-12;
                std::vector<std::vector<unsigned int>> mappings(n_mpi);
                
                int offset = 0;
                for (int r = 0; r < n_mpi; ++r) {
                    mappings[r].resize(sizes[r]);
                    
                    for (int i = 0; i < sizes[r]; ++i) {
                        dealii::Point<dim> p;
                        for (int d = 0; d < dim; ++d) {
                            p[d] = all_coords[offset++];
                        }
                        
                        // Find if vertex already exists (with tolerance)
                        unsigned int global_idx = unique_vertices.size();
                        for (unsigned int j = 0; j < unique_vertices.size(); ++j) {
                            if (unique_vertices[j].distance(p) < tolerance) {
                                global_idx = j;
                                break;
                            }
                        }
                        
                        // Add new vertex if not found
                        if (global_idx == unique_vertices.size()) {
                            unique_vertices.push_back(p);
                        }
                        
                        mappings[r][i] = global_idx;
                    }
                }
                
                pcout << "Vertex deduplication: " << (offset/dim) << " -> " << unique_vertices.size() << std::endl;
                
                // Send mappings back to other ranks
                local_to_global = mappings[0];
                for (int r = 1; r < n_mpi; ++r) {
                    MPI_Send(mappings[r].data(), mappings[r].size(), 
                            MPI_UNSIGNED, r, 0, mpi_communicator);
                }
            } else {
                MPI_Gatherv(local_coords.data(), local_coords.size(), MPI_DOUBLE,
                            nullptr, nullptr, nullptr, MPI_DOUBLE, 0, mpi_communicator);
                local_to_global.resize(n_local);
                MPI_Recv(local_to_global.data(), n_local, MPI_UNSIGNED, 0, 0, 
                        mpi_communicator, MPI_STATUS_IGNORE);
            }

            // --- MPI GATHERING PHASE 2: CELL CONNECTIVITY ---
            
            // Build cell connectivity using corrected vertex mapping
            std::vector<unsigned int> local_cells;
            unsigned int n_local_cells = 0;
            for (auto cell = tria.begin_active(); cell != tria.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
                n_local_cells++;
                for (unsigned int v = 0; v < verts_per_cell; ++v) {
                    unsigned int old_idx = cell->vertex_index(v);
                    unsigned int local_idx = old_to_local_map[old_idx];
                    local_cells.push_back(local_to_global[local_idx]);
                }
            }
            
            // Gather cell counts
            std::vector<int> cell_counts(n_mpi);
            int local_cell_count = static_cast<int>(n_local_cells);
            MPI_Gather(&local_cell_count, 1, MPI_INT, 
                    cell_counts.data(), 1, MPI_INT, 0, mpi_communicator);
            
            // Gather cell connectivity
            std::vector<unsigned int> all_cell_vertex_data;
            unsigned int total_cells = 0;
            
            if (mpi_rank == 0) {
                std::vector<int> cell_recvcounts(n_mpi), cell_displs(n_mpi, 0);
                for (int r = 0; r < n_mpi; ++r) {
                    cell_recvcounts[r] = cell_counts[r] * verts_per_cell;
                    total_cells += cell_counts[r];
                }
                for (int r = 1; r < n_mpi; ++r) {
                    cell_displs[r] = cell_displs[r - 1] + cell_recvcounts[r - 1];
                }
                all_cell_vertex_data.resize(total_cells * verts_per_cell);
                
                MPI_Gatherv(local_cells.data(), local_cells.size(), MPI_UNSIGNED,
                            all_cell_vertex_data.data(), cell_recvcounts.data(), cell_displs.data(),
                            MPI_UNSIGNED, 0, mpi_communicator);
            } else {
                MPI_Gatherv(local_cells.data(), local_cells.size(), MPI_UNSIGNED,
                            nullptr, nullptr, nullptr, MPI_UNSIGNED, 0, mpi_communicator);
            }

            // --- MPI GATHERING PHASE 3: METRIC DATA ---
            
            const std::vector<dealii::Tensor<2, dim>>& original_metrics = 
                metric_data.valid && metric_data.h_field ? 
                metric_data.h_field->get_inverse_metric_vector() : 
                std::vector<dealii::Tensor<2, dim>>();
            
            const unsigned int metric_size = dim * dim;
            
            // Collect local metric data
            std::vector<double> local_cell_metric_data(n_local_cells * metric_size);
            unsigned int idx = 0;
            for (auto cell = tria.begin_active(); cell != tria.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
                
                const unsigned int cell_index = cell->active_cell_index();
                const dealii::Tensor<2, dim>& metric = 
                    (cell_index < original_metrics.size()) ? 
                    original_metrics[cell_index] : 
                    dealii::Tensor<2, dim>(dealii::unit_symmetric_tensor<dim>());
                
                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        local_cell_metric_data[idx++] = metric[i][j];
                    }
                }
            }
            
            // Gather metric data
            std::vector<double> all_cell_metric_data;
            
            if (mpi_rank == 0) {
                std::vector<int> metric_recvcounts(n_mpi), metric_displs(n_mpi, 0);
                for (int r = 0; r < n_mpi; ++r) {
                    metric_recvcounts[r] = cell_counts[r] * metric_size;
                }
                for (int r = 1; r < n_mpi; ++r) {
                    metric_displs[r] = metric_displs[r - 1] + metric_recvcounts[r - 1];
                }
                all_cell_metric_data.resize(total_cells * metric_size);
                
                MPI_Gatherv(local_cell_metric_data.data(), local_cell_metric_data.size(), MPI_DOUBLE,
                            all_cell_metric_data.data(), metric_recvcounts.data(), metric_displs.data(), 
                            MPI_DOUBLE, 0, mpi_communicator);
            } else {
                MPI_Gatherv(local_cell_metric_data.data(), local_cell_metric_data.size(), MPI_DOUBLE,
                            nullptr, nullptr, nullptr, MPI_DOUBLE, 0, mpi_communicator);
            }

            // --- SERIAL OUTPUT ON RANK 0 ---
            if (mpi_rank == 0) {
                // Build CellData structure
                std::vector<dealii::CellData<dim>> cells(total_cells);
                for (unsigned int c = 0; c < total_cells; ++c) {
                    for (unsigned int v = 0; v < verts_per_cell; ++v) {
                        cells[c].vertices[v] = all_cell_vertex_data[c * verts_per_cell + v];
                    }
                    cells[c].material_id = 0;
                }

                // Reconstruct metric tensors
                std::vector<dealii::Tensor<2, dim>> serial_metrics(total_cells);
                for (unsigned int c = 0; c < total_cells; ++c) {
                    unsigned int offset = c * metric_size;
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            serial_metrics[c][i][j] = all_cell_metric_data[offset++];
                        }
                    }
                }

                // Create serial triangulation
                dealii::Triangulation<dim> serial_tria;
                serial_tria.create_triangulation(unique_vertices, cells, dealii::SubCellData());

                // Create serial DoFHandler
                dealii::DoFHandler<dim> serial_dof_handler(serial_tria);
                dealii::FESystem<dim> fe(dealii::FE_DGQ<dim>(flow_solver->dg->max_degree), nstate);
                serial_dof_handler.distribute_dofs(fe);

                // Write mesh and metric field
                std::ofstream out_msh(filename);
                out_msh << std::scientific << std::setprecision(16);
                PHiLiP::GridRefinement::MshOut<dim, double> msh_out(serial_dof_handler);
                msh_out.add_data_vector(serial_metrics, 
                                    PHiLiP::GridRefinement::StorageType::element, 
                                    "metric");
                msh_out.write_msh(out_msh);

                pcout << "Mesh written successfully: " << unique_vertices.size() << " vertices, "
                    << total_cells << " cells" << std::endl;
            }

            MPI_Barrier(mpi_communicator);
        }


        template <int dim, int nstate>
        void LpCVTMeshAdaptationCases<dim, nstate>::extract_shock_nodes_from_msh(
            const std::string& msh_filename,
            const std::string& output_txt_filename,
            const double x_tolerance,
            const double min_distance_between_nodes) const
        {
            pcout << "\nExtracting shock nodes from: " << msh_filename << std::endl;
            

            auto expected_x = [](double y) { return -0.4*y*y + 0.4*y + 0.375; ; };
            
            std::vector<dealii::Point<dim>> shock_nodes;
            
            // Read and parse the .msh file
            std::ifstream msh_file(msh_filename);
            if (!msh_file.is_open()) {
                pcout << "ERROR: Cannot open file " << msh_filename << std::endl;
                return;
            }
            
            std::string line;
            
            // Skip to $Nodes section
            while (std::getline(msh_file, line)) {
                if (line == "$Nodes") break;
            }
            
            // Read node header: numEntityBlocks numNodes minNodeTag maxNodeTag
            std::getline(msh_file, line);
            std::istringstream header_stream(line);
            int num_entity_blocks, num_nodes, min_tag, max_tag;
            header_stream >> num_entity_blocks >> num_nodes >> min_tag >> max_tag;
            
            pcout << "  Total nodes in mesh: " << num_nodes << std::endl;
            
            // Read entity blocks
            for (int block = 0; block < num_entity_blocks; ++block) {
                // Read entity header: entityDim entityTag parametric numNodesInBlock
                std::getline(msh_file, line);
                std::istringstream entity_stream(line);
                int entity_dim, entity_tag, parametric, num_nodes_block;
                entity_stream >> entity_dim >> entity_tag >> parametric >> num_nodes_block;
                
                // Read node tags
                std::vector<int> node_tags(num_nodes_block);
                for (int i = 0; i < num_nodes_block; ++i) {
                    std::getline(msh_file, line);
                    node_tags[i] = std::stoi(line);
                }
                
                // Read node coordinates
                for (int i = 0; i < num_nodes_block; ++i) {
                    std::getline(msh_file, line);
                    std::istringstream coord_stream(line);
                    double x, y, z;
                    coord_stream >> x >> y >> z;
                    
                    // Check if node is near the shock curve
                    const double x_expected = expected_x(y);
                    const double dx = std::abs(x - x_expected);
                    
                    if (dx <= x_tolerance) {
                        shock_nodes.push_back(dealii::Point<dim>(x, y));
                        pcout << "    Found shock node at (" << x << ", " << y 
                            << "), expected_x = " << x_expected 
                            << ", error = " << dx << std::endl;
                    }
                }
            }
            
            msh_file.close();
            
            pcout << "  Found " << shock_nodes.size() << " nodes near shock (before filtering)" << std::endl;
            
            // Remove nodes that are too close to each other
            std::vector<dealii::Point<dim>> filtered_nodes;
            for (const auto& node : shock_nodes) {
                bool too_close = false;
                for (const auto& existing : filtered_nodes) {
                    if (node.distance(existing) < min_distance_between_nodes) {
                        too_close = true;
                        break;
                    }
                }
                if (!too_close) {
                    filtered_nodes.push_back(node);
                }
            }
            
            pcout << "  After filtering: " << filtered_nodes.size() << " nodes" << std::endl;
            
            // Sort nodes by y-coordinate for better organization
            std::sort(filtered_nodes.begin(), filtered_nodes.end(),
                    [](const dealii::Point<dim>& a, const dealii::Point<dim>& b) {
                        return a[1] < b[1];  // Sort by y
                    });
            
            // Write to output file (only on rank 0 for MPI)
            if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
                std::ofstream out_file(output_txt_filename);
                out_file << std::setprecision(16) << std::scientific;
                
                for (const auto& node : filtered_nodes) {
                    out_file << node[0] << " " << node[1] << "\n";
                }
                
                out_file.close();
                pcout << "  Wrote " << filtered_nodes.size() << " shock nodes to " 
                    << output_txt_filename << std::endl;
            }
        }


        /**
         * Main test runner for LpCVT mesh adaptation
         * Implements outer loop: optimize -> extract metric -> LpCVT -> interpolate
         */
        template <int dim, int nstate>
        int LpCVTMeshAdaptationCases<dim, nstate>::run_test() const
        {
            pcout << "Running LpCVT Mesh Adaptation Test" << std::endl;

            // Get test parameters
            const Parameters::AllParameters param = *(TestsBase::all_parameters);
            
            // Get MPI information
            const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
            const int mpi_size = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
            
            pcout << "MPI: Running with " << mpi_size << " processes, this is rank " << mpi_rank << std::endl;

            //Create initial flow solver
            std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver =
                FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);

            // Run initial solve to get baseline solution
            pcout << "\nRunning initial flow solve..." << std::endl;
            flow_solver->run();
            pcout << "Initial solve completed." << std::endl;

            // Outer loop parameters
            const unsigned int max_outer_loops = 8;

            // Main outer loop: alternates between optimization and mesh reconstruction
            for (unsigned int outer_loop = 0; outer_loop < max_outer_loops; ++outer_loop)
            {
                pcout << "\n========================================" << std::endl;
                pcout << "OUTER LOOP " << outer_loop + 1 << "/" << max_outer_loops << std::endl;
                pcout << "========================================" << std::endl;

                // std::array<double, nstate> l2_errors = evaluate_L2_error_norm(flow_solver->dg);
                // pcout << "  L2 Errors at outer loop #" << outer_loop + 1 << " start : ";
                // for (unsigned int s = 0; s < nstate; ++s) {
                //     pcout << l2_errors[s] << " ";
                // }
                // pcout << std::endl;


                // Step 1: Run inner adaptation cycles (mesh optimization)
                std::vector<double> functional_error_vector;
                std::vector<unsigned int> n_cycle_vector;
                std::vector<unsigned int> n_dofs_vector;
                flow_solver->dg->output_results_vtk(outer_loop * 10 + 1, 0.0); 
                // if (outer_loop == 4) {                
                //     bool adaptation_success = run_adaptation_loop(flow_solver, param,
                //     param.mesh_adaptation_param.total_mesh_adaptation_cycles,
                //     functional_error_vector, n_dofs_vector, n_cycle_vector, 
                //     outer_loop + 1, "l2_vs_dofs.txt");

                //     if (!adaptation_success) {
                //         pcout << "\nAdaptation stopped, proceeding to LpCVT reconstruction." << std::endl;
                //     }
                // }
                bool adaptation_success = run_adaptation_loop(flow_solver, param,
                    param.mesh_adaptation_param.total_mesh_adaptation_cycles,
                    functional_error_vector, n_dofs_vector, n_cycle_vector, 
                    outer_loop + 1);

                if (!adaptation_success) {
                    pcout << "\nAdaptation stopped, proceeding to LpCVT reconstruction." << std::endl;
                }
                flow_solver->run(); // Final solve after adaptation
                flow_solver->dg->output_results_vtk(outer_loop * 10 + 2, 0.0);
                // std::array<double, nstate> l2_errors_after = evaluate_L2_error_norm(flow_solver->dg);
                // pcout << "  L2 Errors at outer loop #" << outer_loop + 1 << " end : ";
                // for (unsigned int s = 0; s < nstate; ++s) {
                //     pcout << l2_errors_after[s] << " ";
                // }
                // pcout << std::endl;

                // Step 2: Extract metric field from current solution
                // This computes the anisotropic metric tensor that guides mesh generation
                MetricData metric_data = extract_metric_field(flow_solver, outer_loop, param);

                if (!metric_data.valid) {
                    pcout << "ERROR: Could not extract metric field. Stopping outer loop." << std::endl;
                    break;
                }

                // Step 3: Write background mesh with metric for LpCVT solver
                const std::string background_mesh_filename = "lpcvt_background_mesh.msh";
                const std::string save_background_mesh = "lpcvt_background_mesh_outerloop_" + std::to_string(outer_loop) + ".msh";
                write_lpcvt_background_mesh(background_mesh_filename, flow_solver, metric_data);
                write_lpcvt_background_mesh(save_background_mesh, flow_solver, metric_data);

                const double x_tolerance = 0.01;  // Tolerance for distance from shock curve
                const double min_node_spacing = 0.01;  // Minimum distance between output nodes
                extract_shock_nodes_from_msh(background_mesh_filename, 
                                            "shock_vertices.txt",
                                            x_tolerance,
                                            min_node_spacing);

                // Step 4: Call external LpCVT Python solver
                // This generates a new optimal mesh based on the metric field
                pcout << "\n--- Calling external LpCVT solver ---" << std::endl;

                int lpcvt_result = 0;
                if (mpi_rank == 0) {
                    std::string python_cmd = "python run.py";
                    pcout << "Executing: " << python_cmd << std::endl;
                    lpcvt_result = std::system(python_cmd.c_str());
                }
                
                // Broadcast the result to all ranks
                MPI_Bcast(&lpcvt_result, 1, MPI_INT, 0, mpi_communicator);
                
                if (lpcvt_result != 0) {
                    pcout << "ERROR: LpCVT solver failed with exit code " << lpcvt_result << ". Stopping." << std::endl;
                    break;
                }

                // Ensure all ranks wait for the external process to complete
                MPI_Barrier(mpi_communicator);
                
                pcout << "LpCVT solver completed successfully." << std::endl;

                // Step 5: Interpolate solution from old mesh to new LpCVT mesh
                pcout << "\n--- Interpolating solution to new mesh ---" << std::endl;

                const std::string lpcvt_output_mesh = "final_quad_mesh.msh";
                
                // Check if output file exists (rank 0 checks and broadcasts)
                bool file_exists = false;
                if (mpi_rank == 0) {
                    std::ifstream test_file(lpcvt_output_mesh);
                    file_exists = test_file.good();
                    test_file.close();
                    
                    if (!file_exists) {
                        pcout << "WARNING: LpCVT output file not found, waiting..." << std::endl;
                        // Give file system time to synchronize
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        test_file.open(lpcvt_output_mesh);
                        file_exists = test_file.good();
                        test_file.close();
                    }
                }
                MPI_Bcast(&file_exists, 1, MPI_C_BOOL, 0, mpi_communicator);
                
                if (!file_exists) {
                    pcout << "ERROR: LpCVT output mesh file not found: " << lpcvt_output_mesh << std::endl;
                    break;
                }

                // Create interpolator for solution transfer
                std::ostringstream null_stream;
                std::ostream& out_stream = (pcout.is_active() ? std::cout : null_stream);
                MeshInterpolation<dim, nstate, dealii::parallel::distributed::Triangulation<dim>>
                    interpolator(out_stream);

                // Perform interpolation from old mesh to new mesh
                auto new_dg = interpolator.perform_mesh_interpolation(
                    flow_solver->dg,
                    param,
                    param.flow_solver_param.poly_degree,
                    lpcvt_output_mesh
                );

                if (!new_dg) {
                    pcout << "ERROR: Mesh interpolation failed. Stopping." << std::endl;
                    break;
                }

                // Set boundary IDs for the new mesh (problem-specific)
                // This example assumes a unit square domain [0,1]x[0,1]
                const double tol = 1e-10;  // Tolerance for floating-point comparison
                for (auto cell : new_dg->triangulation->active_cell_iterators()) {
                    if (cell->is_artificial()) continue;
                    if (!cell->is_locally_owned()) continue; // Only process locally owned cells
                    
                    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
                        if (cell->face(f)->at_boundary()) {
                            const dealii::Point<dim> center = cell->face(f)->center();
                            // Assign boundary IDs based on position
                            if (std::abs(center[0] - 0.0) < tol) cell->face(f)->set_boundary_id(3);  // Left (x=0)
                            else if (std::abs(center[0] - 1.0) < tol) cell->face(f)->set_boundary_id(1);  // Right (x=1)
                            else if (std::abs(center[1] - 0.0) < tol) cell->face(f)->set_boundary_id(0);  // Bottom (y=0)
                            else if (std::abs(center[1] - 1.0) < tol) cell->face(f)->set_boundary_id(2);  // Top (y=1)
                        }
                    }
                }

                // Step 6: Create new flow solver with interpolated solution on new mesh
                pcout << "\n--- Setting up flow solver on new mesh ---" << std::endl;

                // // Create fresh flow solver instance
                // flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(
                //     &param, parameter_handler);

                // Replace its DG with our interpolated one
                flow_solver->dg = new_dg;
                flow_solver->ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(new_dg);
                flow_solver->ode_solver->allocate_ode_system();
                flow_solver->dg->high_order_grid->update_mapping_fe_field();
                flow_solver->dg->assemble_residual ();
                // Solve on the new mesh
                pcout << "Solving on new mesh..." << std::endl;

                flow_solver->run();

                
                pcout << "\n--- Ready for next outer loop iteration ---" << std::endl;
            }

            pcout << "\n=== LpCVT Test Completed ===" << std::endl;
            return 0;
        }

        // Explicit instantiation for 2D cases
#if PHILIP_DIM==2
        template class LpCVTMeshAdaptationCases<PHILIP_DIM, 1>;
        template class LpCVTMeshAdaptationCases<PHILIP_DIM, PHILIP_DIM + 2>;
#endif

    } // namespace Tests
} // namespace PHiLiP