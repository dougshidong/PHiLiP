#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_split_inviscid_taylor_green_vortex.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"
#include "mesh/grids/straight_periodic_cube.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreen<dim, nstate>::EulerTaylorGreen(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}
template<int dim, int nstate>
std::array<double,2> EulerTaylorGreen<dim, nstate>::compute_change_in_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> energy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        //get solution modal coeff
        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        //interpolate solution to quadrature points
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //compute entropy and kinetic energy "entropy" variables at quad points
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
            std::array<double,nstate> kin_energy_state = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                    energy_var_at_q[istate].resize(n_quad_pts);
                }
                energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        //project the enrtopy and KE var to modal coefficients
        //then write it into a global vector
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
            std::vector<double> energy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(energy_var_at_q[istate], energy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
                energy_var_hat_global[dofs_indices[idof]] = energy_var_hat[ishape];
            }
        }
    }

    //evaluate the change in entropy and change in KE
    dg->assemble_residual();
    std::array<double,2> change_entropy_and_energy;
    change_entropy_and_energy[0] = entropy_var_hat_global * dg->right_hand_side;
    change_entropy_and_energy[1] = energy_var_hat_global * dg->right_hand_side;
    return change_entropy_and_energy;
}
template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_volume_term(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const unsigned int grid_degree = dg->high_order_grid->fe_system.tensor_degree();

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    soln_basis.build_1D_surface_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_face_quadrature);

    OPERATOR::basis_functions<dim,2*dim,double> flux_basis(1, poly_degree, dg->max_grid_degree);
    flux_basis.build_1D_volume_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    flux_basis.build_1D_gradient_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    flux_basis.build_1D_surface_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_face_quadrature);

    OPERATOR::local_basis_stiffness<dim,2*dim,double> flux_basis_stiffness(1, poly_degree, dg->max_grid_degree, true);
    //flux basis stiffness operator for skew-symmetric form
    flux_basis_stiffness.build_1D_volume_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);
    const std::vector<double> &oneD_vol_quad_weights = dg->oneD_quadrature_collection[poly_degree].get_weights();

    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    double volume_term = 0.0;
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix.
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);



        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        std::array<std::vector<double>,dim> vel_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //get volume entropy var and interp to face
        std::array<std::vector<double>,nstate> energy_var_vol_int;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> energy_var;
            energy_var = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    energy_var_vol_int[istate].resize(n_quad_pts);
                }
                energy_var_vol_int[istate][iquad] = energy_var[istate];
            }
        }

        std::array<std::vector<double>,nstate> energy_var_hat;
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            energy_var_hat[istate].resize(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(energy_var_vol_int[istate], energy_var_hat[istate],
                                                 vol_projection.oneD_vol_operator);
        }
        // The matrix of two-pt fluxes for Hadamard products
        std::array<dealii::Tensor<1,dim,dealii::FullMatrix<double>>,nstate> conv_ref_2pt_flux_at_q;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            //extract soln and auxiliary soln at quad pt to be used in physics
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
            // The way it is stored in metric_operators is to use sum-factorization in each direction,
            // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
            dealii::Tensor<2,dim,double> metric_cofactor;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
                }
            }

            // Evaluate physical convective flux
            // If 2pt flux, transform to reference at construction to improve performance.
            // We technically use a REFERENCE 2pt flux for all entropy stable schemes.
            std::array<dealii::Tensor<1,dim,double>,nstate> conv_phys_flux_2pt;
            std::vector<std::array<dealii::Tensor<1,dim,double>,nstate>> conv_ref_flux_2pt(n_quad_pts);
            for (unsigned int flux_basis=iquad; flux_basis<n_quad_pts; ++flux_basis) {

                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,double> metric_cofactor_flux_basis;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_flux_basis[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][flux_basis];
                    }
                }
                std::array<double,nstate> soln_state_flux_basis;
                for(int istate=0; istate<nstate; istate++){
                    soln_state_flux_basis[istate] = soln_at_q[istate][flux_basis];
                }
                //Compute the physical flux
                conv_phys_flux_2pt = euler_double->convective_numerical_split_flux(soln_state, soln_state_flux_basis);

                //Need to subtract off the pressure average term
                const double pressure_int = euler_double->compute_pressure(soln_state);
                const double pressure_ext = euler_double->compute_pressure(soln_state_flux_basis);
                for(int idim=0; idim<dim; idim++){
                    conv_phys_flux_2pt[1+idim][idim] -= 0.5*(pressure_int + pressure_ext);
                }


                for(int istate=0; istate<nstate; istate++){
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor + metric_cofactor_flux_basis),
                        conv_ref_flux_2pt[flux_basis][istate]);
                }
            }

            //Write the values in a way that we can use sum-factorization on.
            for(int istate=0; istate<nstate; istate++){
                //Write the data in a way that we can use sum-factorization on.
                //Since sum-factorization improves the speed for matrix-vector multiplications,
                //We need the values to have their inner elements be vectors.
                for(int idim=0; idim<dim; idim++){
                    //allocate
                    if(iquad == 0){
                        conv_ref_2pt_flux_at_q[istate][idim].reinit(n_quad_pts, n_quad_pts);
                    }
                    //write data
                    for (unsigned int flux_basis=iquad; flux_basis<n_quad_pts; ++flux_basis) {
                        //Note that the 2pt flux matrix is symmetric so we only computed upper triangular
                        conv_ref_2pt_flux_at_q[istate][idim][iquad][flux_basis] = conv_ref_flux_2pt[flux_basis][istate][idim];
                        conv_ref_2pt_flux_at_q[istate][idim][flux_basis][iquad] = conv_ref_flux_2pt[flux_basis][istate][idim];
                    }
                }
            }
        }
        //For each state we:
        //  1. Compute reference divergence.
        //  2. Then compute and write the rhs for the given state.
        for(int istate=0; istate<nstate; istate++){

            //Compute reference divergence of the reference fluxes.
            std::vector<double> conv_flux_divergence(n_quad_pts);

            //2pt flux Hadamard Product, and then multiply by vector of ones scaled by 1.
            // Same as the volume term in Eq. (15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485. but,
            // where we use the reference skew-symmetric stiffness operator of the flux basis for the Q operator and the reference two-point flux as to make use of Alex's Hadamard product
            // sum-factorization type algorithm that exploits the structure of the flux basis in the reference space to have O(n^{d+1}).
            flux_basis.divergence_two_pt_flux_Hadamard_product(conv_ref_2pt_flux_at_q[istate], conv_flux_divergence, oneD_vol_quad_weights, flux_basis_stiffness.oneD_skew_symm_vol_oper, 1.0);

            // Strong form
            // The right-hand side sends all the term to the side of the source term
            // Therefore,
            // \divergence ( Fconv + Fdiss ) = source
            // has the right-hand side
            // rhs = - \divergence( Fconv + Fdiss ) + source
            // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
            // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
            std::vector<double> rhs(n_shape_fns);

            // Convective
            std::vector<double> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(conv_flux_divergence, ones, rhs, soln_basis.oneD_vol_operator, false, -1.0);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                volume_term += energy_var_hat[istate][ishape] * rhs[ishape];
            }
        }
    }

    return volume_term;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    double entropy_fn = 0.0;
    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(dg->max_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, dg->max_grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            const double density = soln_state[0];
            const double pressure = euler_double->compute_pressure(soln_state);
            const double entropy = log(pressure) - euler_double->gam * log(density);
            const double quadrature_entropy = -density*entropy/euler_double->gamm1;

            entropy_fn += quadrature_entropy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];

        }
    }

    return entropy_fn;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_kinetic_energy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy in the L2-norm (physically relevant)
    int overintegrate = 10 ;
    dealii::QGauss<1> quad_extra(dg->max_degree+1+overintegrate);
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    double total_kinetic_energy = 0;

    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;

       // fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);
        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(dg->max_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, dg->max_grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            const double density = soln_state[0];
            const double quadrature_kinetic_energy =  0.5*(soln_state[1]*soln_state[1]
                                                    + soln_state[2]*soln_state[2]
                                                    + soln_state[3]*soln_state[3])/density;

            total_kinetic_energy += quadrature_kinetic_energy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    //get local CFL
    const unsigned int n_dofs_cell = nstate*pow(poly_degree+1,dim);
    const unsigned int n_quad_pts = pow(poly_degree+1,dim);
    std::vector<dealii::types::global_dof_index> dofs_indices1 (n_dofs_cell);

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices (dofs_indices1);
        std::vector< std::array<double,nstate>> soln_at_q(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate=0; istate<nstate; istate++) {
                soln_at_q[iquad][istate]      = 0;
            }
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
          dealii::Point<dim> qpoint = dg->volume_quadrature_collection[poly_degree].point(iquad);
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                soln_at_q[iquad][istate] += dg->solution[dofs_indices1[idof]] * dg->fe_collection[poly_degree].shape_value_component(idof, qpoint, istate);
            }
        }

        std::vector< double > convective_eigenvalues(n_quad_pts);
        for (unsigned int isol = 0; isol < n_quad_pts; ++isol) {
            convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        }
        const double max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));

        double cfl = 0.1 * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    return cfl_min;
}

template <int dim, int nstate>
int EulerTaylorGreen<dim, nstate>::run_test() const
{
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2 * dealii::numbers::PI;
    const int n_refinements = 2;
    unsigned int poly_degree = 3;

    const unsigned int grid_degree = all_parameters->use_curvilinear_grid ? poly_degree : 1;
    if(all_parameters->use_curvilinear_grid){
        //if curvilinear
        PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);
    }
    else{
        //if straight
        PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, pow(2.0,n_refinements));
    }

    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    pcout << "Implement initial conditions" << std::endl;
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/pow(n_global_active_cells2,1.0/dim)/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    pcout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    pcout << "ODE solver successfully created" << std::endl;
    double finalTime = 14.;

    pcout << " number dofs " << dg->dof_handler.n_dofs()<<std::endl;
    pcout << "preparing to advance solution in time" << std::endl;

    ode_solver->current_iteration = 0;
    ode_solver->allocate_ode_system();

    const double initial_energy = compute_kinetic_energy(dg, poly_degree);
    const double initial_energy_mpi = (dealii::Utilities::MPI::sum(initial_energy, mpi_communicator));
    const double initial_entropy = compute_entropy(dg, poly_degree);
    const double initial_entropy_mpi = (dealii::Utilities::MPI::sum(initial_entropy, mpi_communicator));
    //create a file to wirte entorpy and energy results to
    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);
    //loop over time
    while(ode_solver->current_time < finalTime){
        //get timestep
        const double time_step =  get_timestep(dg,poly_degree, delta_x);
        if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
            pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
        //take the minimum timestep from all processors.
        const double dt = dealii::Utilities::MPI::min(time_step, mpi_communicator);
        //integrate in time
        ode_solver->step_in_time(dt, false);
        ode_solver->current_iteration += 1;
        //check if print solution
        const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
        if (is_output_iteration) {
            const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
            dg->output_results_vtk(file_number);
        }

        //get the change in entropy
        const std::array<double,2> current_change_entropy = compute_change_in_entropy(dg, poly_degree);
        const double current_change_entropy_mpi = dealii::Utilities::MPI::sum(current_change_entropy[0], mpi_communicator);
        const double current_change_energy_mpi = dealii::Utilities::MPI::sum(current_change_entropy[1], mpi_communicator);
        //write to the file the change in entropy mpi
        myfile<<ode_solver->current_time<<" "<< current_change_entropy_mpi <<std::endl;
        pcout << "M plus K norm Change in Entropy at time " << ode_solver->current_time << " is " << current_change_entropy_mpi<< std::endl;
        pcout << "M plus K norm Change in Kinetic Energy at time " << ode_solver->current_time << " is " << current_change_energy_mpi<< std::endl;
        //check if change in entropy is conserved at machine precision
        if(abs(current_change_entropy[0]) > 1e-12 && (dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::IR || dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::CH || dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::Ra)){
          pcout << " Change in entropy was not monotonically conserved." << std::endl;
          return 1;
        }

        //get the kinetic energy
        const double current_energy = compute_kinetic_energy(dg, poly_degree);
        const double current_energy_mpi = (dealii::Utilities::MPI::sum(current_energy, mpi_communicator));
        pcout << "Normalized kinetic energy " << ode_solver->current_time << " is " << current_energy_mpi/initial_energy_mpi<< std::endl;
        //get the entropy
        const double current_entropy = compute_entropy(dg, poly_degree);
        const double current_entropy_mpi = (dealii::Utilities::MPI::sum(current_entropy, mpi_communicator));
        pcout << "Normalized entropy " << ode_solver->current_time << " is " << current_entropy_mpi/initial_entropy_mpi<< std::endl;

        //get the volume work for kinetic energy
        double current_vol_work = compute_volume_term(dg, poly_degree);
        double current_vol_work_mpi = (dealii::Utilities::MPI::sum(current_vol_work, mpi_communicator));
        pcout<<"volume work "<<current_vol_work_mpi<<std::endl;
        myfile << ode_solver->current_time << " " << std::fixed << std::setprecision(16) << current_vol_work_mpi<< std::endl;
        //check for non overintegrated case
        if(!all_parameters->use_curvilinear_grid and all_parameters->overintegration == 0){
            if(abs(current_vol_work_mpi) > 1e-12 && (dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::Ra || dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::KG ) ){
                pcout<< "The kinetic energy volume work is not zero."<<std::endl;
                return 1;
            }
        }
    }
    myfile.close();

    return 0;
}

#if PHILIP_DIM==3
    template class EulerTaylorGreen <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

