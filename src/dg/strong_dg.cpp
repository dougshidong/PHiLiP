#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <Sacado.hpp>
//#include <deal.II/differentiation/ad/sacado_math.h>
//#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

#include "dg.h"
#include "physics/physics_factory.h"

namespace PHiLiP {

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    template <int dim> using Triangulation = dealii::Triangulation<dim>;
#else
    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif


template <int dim, int nstate, typename real>
DGStrong<dim,nstate,real>::DGStrong(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    Triangulation *const triangulation_input)
    : DGBase<dim,real>::DGBase(nstate, parameters_input, degree, triangulation_input) // Use DGBase constructor
{
    using ADtype = Sacado::Fad::DFad<real>;
    pde_physics = Physics::PhysicsFactory<dim,nstate,ADtype> ::create_Physics(parameters_input);
    conv_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics);
    diss_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics);

    pde_physics_double = Physics::PhysicsFactory<dim,nstate,real> ::create_Physics(parameters_input);
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics_double);
}

template <int dim, int nstate, typename real>
DGStrong<dim,nstate,real>::~DGStrong ()
{ 
    pcout << "Destructing DGStrong..." << std::endl;
    delete conv_num_flux;
    delete diss_num_flux;
}

/********************************************************************
 * Get projection operator for derivative
 * *******************************************************************/
//#if 0
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::get_projection_operator(
                const dealii::FEValues<dim,dim> &fe_values_volume, unsigned int n_quad_pts,
                 unsigned int n_dofs_cell, dealii::FullMatrix<real> &projection_matrix)
{


        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_vandermonde(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> local_weights(n_quad_pts);
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_volume.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe_values_volume.shape_value_component(itest,iquad,istate_test)
                        * fe_values_volume.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_volume.JxW(iquad);
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        local_vandermonde[iquad][itest] = fe_values_volume.shape_value_component(itest,iquad,istate_test);
                }
        }
        for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for( unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                local_weights[iquad][iquad2] = 0.0;
                if(iquad==iquad2)
                    local_weights[iquad][iquad2] = fe_values_volume.JxW(iquad);
            }
        }

        local_inverse_mass_matrix.invert(local_mass_matrix);
        dealii::FullMatrix<real> V_trans_W(n_dofs_cell, n_quad_pts);

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                V_trans_W[itest][iquad] = local_vandermonde[iquad][itest] * fe_values_volume.JxW(iquad);
            }
        }


    for (unsigned int idof=0; idof<n_dofs_cell; idof++){
        for (unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            projection_matrix[idof][iquad] = 0.0;
            for (unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                projection_matrix[idof][iquad] += local_inverse_mass_matrix[idof][idof2] * V_trans_W[idof2][iquad];
            }
        }
    } 

#if 0
    printf("other Proj oper\n");
        fflush(stdout);
    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            printf("%g ",projection_matrix[idof][iquad]);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
#endif


}
//#endif
/************************************************************************/
#if 0
template <int dim, typename real>
dealii::FullMatrix<double> get_projection_operator(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Quadrature<dim> &quad)
{
    const std::vector<dealii::Point<dim>> &quad_pts = quad.get_points();
    const std::vector<double> &quad_weights = quad.get_weights();
    const unsigned int n_quad_pts = quad.size();
    const unsigned int n_dofs = fe.n_dofs_per_cell();

    dealii::FullMatrix<real> local_mass_matrix(n_dofs);
    dealii::FullMatrix<real> local_vandermonde_transpose_weights_matrix(n_dofs, n_quad_pts);
    for (unsigned int itest=0; itest<n_dofs; ++itest) {
        const unsigned int istate_test = fe.system_to_component_index(itest).first;
        for (unsigned int itrial=itest; itrial<n_dofs; ++itrial) {
            const unsigned int istate_trial = fe.system_to_component_index(itrial).first;

            real value = 0.0;
            if (istate_test==istate_trial) { 
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe.shape_value_component(itest,quad_pts[iquad],istate_test)
                        * fe.shape_value_component(itrial,quad_pts[iquad],istate_trial)
                        * quad_weights[iquad];
                }
                //std::cout << "non zero " << value << std::endl;
            }
            //std::cout << value << std::endl;
            local_mass_matrix[itrial][itest] = value;
            local_mass_matrix[itest][itrial] = value;
        }

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            local_vandermonde_transpose_weights_matrix[itest][iquad] = fe.shape_value_component(itest,quad_pts[iquad],istate_test) * quad_weights[iquad];
        }
    }
    
    dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs);
    local_inverse_mass_matrix.invert(local_mass_matrix);

    dealii::FullMatrix<real> projection_operator(n_dofs, n_quad_pts);
    local_inverse_mass_matrix.mmult(projection_operator, local_vandermonde_transpose_weights_matrix);

    return projection_operator;
}
#endif
/*********************************
 * get flux with metric terms
 * *****************************/
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::get_flux_with_metric_terms(
                const dealii::FEValues<dim,dim> &fe_values_vol,
                const unsigned int n_quad_pts, const unsigned int n_dofs_cell,
                const std::vector< std::array< dealii::Tensor<1,dim,real>, nstate >> &conv_phys_flux_at_q, 
                std::vector< std::array< dealii::Tensor<1,dim,real>, nstate >> &conv_phys_flux_metric)
{

    std::vector<dealii::FullMatrix<real>> Jacobian_inv(n_quad_pts);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
       Jacobian_inv[iquad].reinit(dim, dim);
    }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            dealii::DerivativeForm<1, dim, dim> temp;
            temp=fe_values_vol.inverse_jacobian(iquad);
            for(int idim=0; idim<dim; idim++){
                for(int idim2=0; idim2<dim; idim2++){
                    Jacobian_inv[iquad][idim][idim2] = temp[idim][idim2];
                }
            }
        }
        
    const unsigned int fe_index_curr_cell = pow(n_dofs_cell,1.0/dim) - 1;
    const std::vector<real> &quad_weights = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].get_weights ();
    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                conv_phys_flux_metric[iquad][istate][idim] = 0.0;
                for(int idim2=0; idim2<dim; idim2++){
                //conv_phys_flux_metric[iquad][istate][idim] += Jacobian_inv[iquad][idim][idim2] * conv_phys_flux_at_q[iquad][istate][idim];
                conv_phys_flux_metric[iquad][istate][idim] += Jacobian_inv[iquad][idim][idim2] * conv_phys_flux_at_q[iquad][istate][idim] * JxW[iquad] / quad_weights[iquad];
                }
            }
        }
    }

}
/**************************************************************************/
/**************************************************
 * Flux Reconstruction should not be applied to the source term
 * ************************************************/
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::get_Flux_Reconstruction_modifying_filters(
        const dealii::FEValues<dim,dim> &fe_values_vol, const unsigned int n_dofs_cell, 
        const unsigned int n_quad_pts, const dealii::FEValues<dim,dim> &fe_values_vol_soln_flux, 
        const unsigned int n_quad_pts_flux, const dealii::FullMatrix<real> K_operator,
        dealii::FullMatrix<real> &filter_chi, dealii::FullMatrix<real> &filter_chi_source)
{
//ESFR Classical
    dealii::FullMatrix<real> Mass_Jac(n_dofs_cell);
    dealii::FullMatrix<real> M_K(n_dofs_cell);
    dealii::FullMatrix<real> M_inv(n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator_with_Jac_Quad(n_quad_pts, n_dofs_cell);
    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            Chi_operator[iquad][itest] = fe_values_vol.shape_value_component(itest,iquad,istate);
            Chi_operator_with_Jac_Quad[iquad][itest] = fe_values_vol.shape_value_component(itest,iquad,istate) * JxW[iquad];
        }
    }
    }
    Chi_operator.Tmmult(Mass_Jac, Chi_operator_with_Jac_Quad); 
        for(unsigned int iquad=0; iquad<n_dofs_cell; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            M_K[iquad][idof] = Mass_Jac[iquad][idof] + K_operator[iquad][idof];
            }
        }
        M_inv.invert(Mass_Jac);
        dealii::FullMatrix<real> filter(n_dofs_cell);
        M_K.mmult(filter, M_inv);
        dealii::FullMatrix<real> Chi_operator_soln_flux(n_quad_pts_flux,n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
                Chi_operator_soln_flux[iquad][idof] = fe_values_vol_soln_flux.shape_value_component(idof,iquad,0);
            }
        }
        if (this->all_parameters->use_classical_FR == true){
            filter.mTmult(filter_chi, Chi_operator_soln_flux);
        }
        else{
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                for(unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
                    filter_chi[idof][iquad] = Chi_operator_soln_flux[iquad][idof];//filter_chi = Chi^T
                }
            }
        //    filter_chi = Chi_operator_soln_flux;
        }
        filter.mTmult(filter_chi_source, Chi_operator);

//end of ESFR Classical

}
/**************************************************
 * IMPLICIT
 * ************************************************/

/**************************************************************************************/
/***********
 * Auxiliary
 * *************************************/
//compute the volume integrals for convection/diffusion auxiliary equation
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_volume_terms_auxiliary_equation(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS)
{

   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();


    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
    //    soln_coeff[idof].diff(idof, n_dofs_cell);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }

       // std::cout<< "AUX vol soln q: "<< soln_at_q[iquad][0] <<std::endl;

        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
       // diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
    }

#if 0
    const unsigned int n_rows_dif_order_cells = DGBase<dim,real>::dif_order_cells.size() / 4;
    unsigned int idif_order_cell =0;
    for(; idif_order_cell<n_rows_dif_order_cells; idif_order_cell++){
        if(n_dofs_cell == DGBase<dim,real>::dif_order_cells[idif_order_cell][0]
            && n_quad_pts == DGBase<dim,real>::dif_order_cells[idif_order_cell][1])
            break;
    }

    dealii::FullMatrix<real> projection_matrix(n_dofs_cell, n_quad_pts);
    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
        for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const unsigned int dof_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][2];
            const unsigned int quad_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][3];
            projection_matrix[idof][iquad] = DGBase<dim,real>::global_projection_operator[dof_flux_start + idof][quad_flux_start + iquad];
        }
    }

  //  get_projection_operator(fe_values_vol_flux, n_quad_pts_flux, n_dofs_cell_flux, projection_matrix);
    std::vector< ADArray > soln_projected(n_dofs_cell);
    for (unsigned int idof=0; idof<n_dofs_cell; idof++){
       // for (int istate=0; istate<nstate; istate++){
            const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
            soln_projected[idof][istate] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                soln_projected[idof][istate] += projection_matrix[idof][iquad] * soln_at_q[iquad][istate];
            }
       // }
    } 


//get derivative of nonlinear flux
    std::vector< ADArrayTensor1 > soln_div_at_q(n_quad_pts); // Tensor initialize with zeros
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            soln_div_at_q[iquad][istate] = 0.0;
            for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                soln_div_at_q[iquad][istate] += soln_projected[idof][istate] * fe_values_vol.shape_grad_component(idof, iquad, istate);
            }
        diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q[iquad], soln_div_at_q[iquad]);
        }
    }
#endif
//Testing KD
#if 0
    std::vector<dealii::FullMatrix<real>> local_derivative_operator(dim);
    for(int idim=0; idim<dim; idim++){
       local_derivative_operator[idim].reinit(n_quad_pts, n_dofs_cell);
    }
    dealii::FullMatrix<real> Chi_operator(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator_with_Jac(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_inv_operator(n_quad_pts, n_dofs_cell);
    const std::vector<real> &quad_weights = DGBase<dim,real>::volume_quadrature_collection[2].get_weights ();
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection[2].point(iquad);
            Chi_operator[iquad][itest] = DGBase<dim,real>::fe_collection[2].shape_value_component(itest,qpoint,istate);
            Chi_operator_with_Jac[iquad][itest] = DGBase<dim,real>::fe_collection[2].shape_value_component(itest,qpoint,istate) * JxW[iquad] / quad_weights[iquad];
        }
    }
    }
    Chi_inv_operator.invert(Chi_operator);
    dealii::FullMatrix<real> Jacobian_physical(n_dofs_cell);
    Chi_inv_operator.mmult(Jacobian_physical, Chi_operator_with_Jac);
//printf("deriv\n");
//fflush(stdout);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
           // derivative = fe_values_vol.shape_grad_component(idof, iquad, istate);
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection[2].point(iquad);
            derivative = DGBase<dim,real>::fe_collection[2].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                local_derivative_operator[idim][iquad][idof] = derivative[idim];
//printf("%g ",local_derivative_operator[idim][iquad][idof]);
//fflush(stdout);
            }
        }
//printf("\n");
//fflush(stdout);
    }
    }
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> derivative_temp(n_quad_pts, n_dofs_cell);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                derivative_temp[iquad][idof] = local_derivative_operator[idim][iquad][idof];
            }
        }
        Chi_inv_operator.mmult(local_derivative_operator[idim],derivative_temp);
    }

        dealii::FullMatrix<real> derivative_p2(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> derivative_p3(n_quad_pts, n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    derivative_p3[idof][iquad] = 0.0;//set it equal to identity
            }
        }
        for(unsigned int v_deg=0; v_deg<=0; v_deg++){
        dealii::FullMatrix<real> derivative_p(n_quad_pts, n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(idof == iquad){
                    derivative_p[idof][iquad] = 1.0;//set it equal to identity
                }
            }
        }
        
        for(unsigned int idegree=0; idegree< (2 -v_deg); idegree++){
            dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
            derivative_p_temp.add(1, derivative_p);
            local_derivative_operator[0].mmult(derivative_p, derivative_p_temp);
        }
        for(unsigned int idegree=0; idegree< (v_deg); idegree++){
            dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
            derivative_p_temp.add(1, derivative_p);
            local_derivative_operator[1].mmult(derivative_p, derivative_p_temp);
        }
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    derivative_p3[idof][iquad] += derivative_p[idof][iquad];
                    if(v_deg == 1)
                        derivative_p3[idof][iquad] += derivative_p[idof][iquad];
            }
        }

        }
        dealii::FullMatrix<real> derivative_p4(n_quad_pts, n_dofs_cell);
        Chi_operator.mmult(derivative_p4, derivative_p3);
        derivative_p4.mmult(derivative_p2,Jacobian_physical);

    std::vector<ADArrayTensor1> flux_divergence_test(n_dofs_cell);
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
    flux_divergence_test[idof][0] = 0.0;
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            flux_divergence_test[idof][istate] -= fe_values_vol.shape_value_component(idof,iquad,istate) * diss_phys_flux_at_q[iquad][istate]* JxW[iquad];
    }
    }

        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_vol.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_vol.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe_values_vol.shape_value_component(itest,iquad,istate_test)
                        * fe_values_vol.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_vol.JxW(iquad);
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }
            dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
            local_inverse_mass_matrix.invert(local_mass_matrix);

    std::vector<ADArrayTensor1> flux_divergence_test2(n_dofs_cell);
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
    flux_divergence_test2[idof][0] = 0.0;
        for (unsigned int idof2=0; idof2<n_dofs_cell; ++idof2) {
            flux_divergence_test2[idof][istate] += local_inverse_mass_matrix[idof][idof2] * flux_divergence_test[idof2][istate];
    }
    }

    std::vector<ADArrayTensor1> flux_divergence_fin(n_quad_pts);
    for(int idim=0; idim<dim; idim++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    flux_divergence_fin[iquad][0] = 0.0;
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
            flux_divergence_fin[iquad][istate][idim] += derivative_p2[iquad][idof] * flux_divergence_test2[idof][istate][idim];
    }
    //    printf("KD %g\n",flux_divergence_fin[iquad][0]);
     //   fflush(stdout);
    std::cout << "AUXILIARY KD: " << flux_divergence_fin[iquad][0][idim] << "for dim: " << idim << ".\n";
    }
    }
#endif
    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
    
   // const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();

    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        dealii::Tensor<1,dim,ADtype> rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

    //        assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);

            rhs =  rhs - fe_values_vol.shape_value_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
        }

        for (int iDim=0; iDim<dim; iDim++)
        {
           // local_auxiliary_RHS[itest][iDim] += rhs[iDim].val(); 
            local_auxiliary_RHS[itest][iDim] += rhs[iDim];
        }

    #if 0
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(cell_dofs_indices[itest], cell_dofs_indices, residual_derivatives);
        }
        #endif
    }
}
/**************************************************************************************/
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_boundary_term_auxiliary_equation(
    const unsigned int boundary_id, 
    const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS)
{
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();


    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // D(u*-u_int)
    std::vector<ADArray> diss_primary_num_flux_dot_n(n_face_quad_pts); // 

    std::vector<ADArrayTensor1> diss_phys_flux(n_face_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    //const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    //    soln_coeff_int[idof].diff(idof, n_total_indep);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];
    //    const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

       // std::cout<< "AUX boundary soln q: "<< soln_int[iquad][0] <<std::endl;
        const dealii::Point<dim, real> x_quad = quad_pts[iquad];
       // pde_physics->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
        pde_physics_double->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        //
      //  conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        // Used for strong form

        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
       // diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = diss_num_flux_double->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;//(u*-u)*n
        }
       // diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);//-D*(u*-u)*n
        diss_flux_jump_int[iquad] = pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);//-D*(u*-u)*n

    }

    // Boundary integral
        //ADArrayTensor1 rhs;
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        
        //ADArrayTensor1 rhs;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************


        for (int idim=0; idim<dim; idim ++)
        {
           // local_auxiliary_RHS[itest][idim] += rhs[idim].val();   
            local_auxiliary_RHS[itest][idim] += rhs[idim];   
        }
    }
}
/*********************************************************************************/
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_face_term_auxiliary(
    const dealii::FEValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValues<dim,dim>     &fe_values_ext,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    std::vector<dealii::Tensor<1,dim,real>>          &local_rhs_int_cell,
    std::vector<dealii::Tensor<1,dim,real>>          &local_rhs_ext_cell)
{
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
   // const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    //    soln_coeff_int_ad[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
     //   soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_ext[iquad][istate]      = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]   += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]   += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
        }

       // std::cout<< "AUX face soln q int: "<< soln_int[iquad][0] <<std::endl;
       // std::cout<< "AUX face soln q ext: "<< soln_ext[iquad][0] <<std::endl;

        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term


       // diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext;
        }
       // diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
       // diss_flux_jump_ext[iquad] = pde_physics->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);
        diss_flux_jump_int[iquad] = pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

        for (int iDim = 0; iDim<dim; iDim++)
        {
           // local_rhs_int_cell[itest_int][iDim] += rhs[iDim].val();
            local_rhs_int_cell[itest_int][iDim] += rhs[iDim];
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

        for (int iDim = 0; iDim<dim; iDim++)
        {
           // local_rhs_ext_cell[itest_ext][iDim] += rhs[iDim].val();
            local_rhs_ext_cell[itest_ext][iDim] += rhs[iDim];
        }
    }
}
/**************************************************************************************/
/********************************
 * Primary Eqn
 * ******************************************/
template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_volume_terms_implicit(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FEValues<dim,dim> &fe_values_vol_flux,
    const dealii::FEValues<dim,dim> &fe_values_vol_soln_flux,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{

    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;
    //FOR FLUX POINTS
    const unsigned int n_quad_pts_flux      = fe_values_vol_flux.n_quadrature_points;
    const unsigned int n_dofs_cell_flux     = fe_values_vol_flux.dofs_per_cell;


    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    const std::vector<real> &JxW_soln_flux = fe_values_vol_soln_flux.get_JxW_values ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > aux_soln_at_q(n_quad_pts_flux);
    std::vector< ADArray > aux_soln_div_at_q(n_quad_pts_flux); // Tensor initialize with zeros

    std::vector< ADArray > source_at_q(n_quad_pts);

    //Evaluated at the flux points
    std::vector< ADArray > soln_at_q_flux(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > soln_grad_at_q_flux(n_quad_pts_flux); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > soln_at_q_split(n_quad_pts_flux);


    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
        soln_coeff[idof].diff(idof, n_dofs_cell);
        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff[idof][idim] = auxiliary_solution[idim][cell_dofs_indices[idof]];
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the flux volume quadrature points
            soln_at_q_flux[iquad][istate]      = 0;
            soln_grad_at_q_flux[iquad][istate] = 0;
            aux_soln_at_q[iquad][istate]  = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
           // source_at_q[iquad] = pde_physics->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad]);
            source_at_q[iquad] = pde_physics->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad], DGBase<dim,real>::current_time);
        }
    }
   //flux evaluated at flux points 
    for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q_flux[iquad][istate]      += soln_coeff[idof] * fe_values_vol_soln_flux.shape_value_component(idof, iquad, istate);
              soln_grad_at_q_flux[iquad][istate] += soln_coeff[idof] * fe_values_vol_soln_flux.shape_grad_component(idof, iquad, istate);

              aux_soln_at_q[iquad][istate]  += aux_soln_coeff[idof] * fe_values_vol_soln_flux.shape_value_component(idof, iquad, istate);
        }
        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = pde_physics->convective_flux (soln_at_q_flux[iquad]);
       // diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q_flux[iquad], soln_grad_at_q_flux[iquad]);
        for (int idim=0; idim<dim; idim++){
            for (int istate =0; istate<nstate; istate++){
                soln_at_q_split[iquad][istate][idim] = soln_at_q_flux[iquad][istate];
            }
        }

    }
//build projection for differentiation
    
    const unsigned int n_rows_dif_order_cells = DGBase<dim,real>::dif_order_cells.size() / 4;
    unsigned int idif_order_cell =0;
    for(; idif_order_cell<n_rows_dif_order_cells; idif_order_cell++){
        if(n_dofs_cell == DGBase<dim,real>::dif_order_cells[idif_order_cell][0]
            && n_quad_pts == DGBase<dim,real>::dif_order_cells[idif_order_cell][1])
            break;
    }

    dealii::FullMatrix<real> projection_matrix(n_dofs_cell_flux, n_quad_pts_flux);
    for(unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
        for( unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
            const unsigned int dof_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][2];
            const unsigned int quad_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][3];
            projection_matrix[idof][iquad] = DGBase<dim,real>::global_projection_operator[dof_flux_start + idof][quad_flux_start + iquad];
        }
    }

  //  get_projection_operator(fe_values_vol_flux, n_quad_pts_flux, n_dofs_cell_flux, projection_matrix);
    std::vector< ADArrayTensor1 > conv_projected_phys_flux(n_dofs_cell_flux);
    std::vector< ADArrayTensor1 > aux_soln_projected(n_dofs_cell_flux);
    std::vector< ADArrayTensor1 > soln_at_q_projected(n_dofs_cell_flux);
    for (unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
       // for (int istate=0; istate<nstate; istate++){
            const unsigned int istate = fe_values_vol_flux.get_fe().system_to_component_index(idof).first;
            conv_projected_phys_flux[idof][istate] = 0.0;
            aux_soln_projected[idof][istate] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
                conv_projected_phys_flux[idof][istate] += projection_matrix[idof][iquad] * conv_phys_flux_at_q[iquad][istate];
                aux_soln_projected[idof][istate] += projection_matrix[idof][iquad] * aux_soln_at_q[iquad][istate];
                soln_at_q_projected[idof][istate] += projection_matrix[idof][iquad] * soln_at_q_split[iquad][istate];
            }
       // }
    } 


//get derivative of nonlinear flux
    std::vector<ADArray> flux_divergence(n_quad_pts_flux);
    std::vector<ADArray> split_div(n_quad_pts_flux);//d(Chi)/d(xi)*u
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
            aux_soln_div_at_q[iquad][istate] = 0.0;
            for (unsigned int idof=0; idof<n_dofs_cell_flux; ++idof) {
                flux_divergence[iquad][istate] += conv_projected_phys_flux[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
                aux_soln_div_at_q[iquad][istate] += aux_soln_projected[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
                split_div[iquad][istate] += soln_at_q_projected[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
            }
        }
    }

//Testing KD
#if 0
    std::vector<dealii::FullMatrix<real>> local_derivative_operator(dim);
    for(int idim=0; idim<dim; idim++){
       local_derivative_operator[idim].reinit(n_quad_pts, n_dofs_cell);
    }
    dealii::FullMatrix<real> Chi_operator(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator_with_Jac(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_inv_operator(n_quad_pts, n_dofs_cell);
    const std::vector<real> &quad_weights = DGBase<dim,real>::volume_quadrature_collection[2].get_weights ();
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection[2].point(iquad);
            Chi_operator[iquad][itest] = DGBase<dim,real>::fe_collection[2].shape_value_component(itest,qpoint,istate);
            Chi_operator_with_Jac[iquad][itest] = DGBase<dim,real>::fe_collection[2].shape_value_component(itest,qpoint,istate) * JxW[iquad] / quad_weights[iquad];
        }
    }
    }
    Chi_inv_operator.invert(Chi_operator);
    dealii::FullMatrix<real> Jacobian_physical(n_dofs_cell);
    Chi_inv_operator.mmult(Jacobian_physical, Chi_operator_with_Jac);
//printf("deriv\n");
//fflush(stdout);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
           // derivative = fe_values_vol.shape_grad_component(idof, iquad, istate);
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection[2].point(iquad);
            derivative = DGBase<dim,real>::fe_collection[2].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                local_derivative_operator[idim][iquad][idof] = derivative[idim];
//printf("%g ",local_derivative_operator[idim][iquad][idof]);
//fflush(stdout);
            }
        }
//printf("\n");
//fflush(stdout);
    }
    }
        dealii::FullMatrix<real> derivative_p(n_quad_pts, n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(idof == iquad){
                    derivative_p[idof][iquad] = 1.0;//set it equal to identity
                }
            }
        }
        
        for(unsigned int idegree=0; idegree< (2); idegree++){
            dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
            derivative_p_temp.add(1, derivative_p);
            local_derivative_operator[0].mmult(derivative_p, derivative_p_temp);
        }
#if 0
printf("Dsquared\n");
fflush(stdout);
for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
for(unsigned int idof=0; idof<n_dofs_cell;idof++){
printf("%g ",derivative_p[iquad][idof]);
fflush(stdout);
}
printf("\n");
fflush(stdout);
}
#endif
        derivative_p.mmult(derivative_p,Jacobian_physical);

    std::vector<ADArray> flux_divergence_test(n_dofs_cell);
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
    flux_divergence_test[idof][0] = 0.0;
        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
            flux_divergence_test[idof][istate] -= fe_values_vol_soln_flux.shape_value_component(idof,iquad,istate) * flux_divergence[iquad][istate]* JxW_soln_flux[iquad];
    }
    }

        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_vol.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_vol.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe_values_vol.shape_value_component(itest,iquad,istate_test)
                        * fe_values_vol.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_vol.JxW(iquad);
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }
            dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
            local_inverse_mass_matrix.invert(local_mass_matrix);

    std::vector<ADArray> flux_divergence_test2(n_dofs_cell);
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
    flux_divergence_test2[idof][0] = 0.0;
        for (unsigned int idof2=0; idof2<n_dofs_cell; ++idof2) {
            flux_divergence_test2[idof][istate] += local_inverse_mass_matrix[idof][idof2] * flux_divergence_test[idof2][istate];
    }
    }

    std::vector<ADArray> flux_divergence_fin(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    flux_divergence_fin[iquad][0] = 0.0;
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof){
        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
            flux_divergence_fin[iquad][istate] += derivative_p[iquad][idof] * flux_divergence_test2[idof][istate];
    }
    //    printf("KD %g\n",flux_divergence_fin[iquad][0]);
     //   fflush(stdout);
   // std::cout << "KD: " << flux_divergence_fin[iquad][0] << ".\n";
    }
#endif

    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0;
        ADtype split = 0;

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {

            // Convective
            // Now minus such 2 integrations by parts
          //  assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);
            double alpha = 1.0;
            if (this->all_parameters->use_split_form == true){
                alpha = 2.0/3.0;
                //for split form
               // split = split + (1.0 - alpha) *  fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * split_div[iquad][istate] * JxW_soln_flux[iquad];
                split = split + (1.0 - alpha) *soln_at_q_flux[iquad][istate] *  fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * split_div[iquad][istate] * JxW_soln_flux[iquad];
            }

            rhs = rhs - alpha * fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];


            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
           // rhs = rhs + fe_values_vol_soln_flux.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW_soln_flux[iquad];
            rhs = rhs + fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * aux_soln_div_at_q[iquad][istate] * JxW_soln_flux[iquad];

            // Source

            if(iquad < n_quad_pts){
                if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                    rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
                }
            }
        }

        if (this->all_parameters->use_split_form == true){
           // split = split * soln_at_q[itest][istate]; 
            rhs = rhs - split;
        }

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(cell_dofs_indices[itest], cell_dofs_indices, residual_derivatives);
        }
    }
}



template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_boundary_term_implicit(
    const unsigned int boundary_id,
    const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> aux_soln_int(n_face_quad_pts);

    std::vector<ADArray> soln_ext(n_face_quad_pts);
    std::vector<ADArrayTensor1> aux_soln_ext(n_face_quad_pts);

    std::vector<ADArray> aux_soln_on_boundary(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
        soln_coeff_int[idof].diff(idof, n_total_indep);
        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff_int[idof][idim] = auxiliary_solution[idim][dof_indices_int[idof]];
        }
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            aux_soln_int[iquad][istate]      = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
            aux_soln_int[iquad][istate]  += aux_soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> x_quad = quad_pts[iquad];
        pde_physics->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        // Used for strong form
        // Which physical convective flux to use?
        conv_phys_flux[iquad] = pde_physics->convective_flux (soln_int[iquad]);

        #if 0
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
        #endif
        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);

        for (int s=0; s<nstate; s++) {
            aux_soln_on_boundary[iquad][s] = (aux_soln_int[iquad][s]) * normal_int;
        }
    }

    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
           // rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) *  aux_soln_on_boundary[iquad][istate] *  JxW[iquad];
        }
        // *******************

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(dof_indices_int[itest], dof_indices_int, residual_derivatives);
        }
    }
}

template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_face_term_implicit(
    const dealii::FEValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValues<dim,dim>     &fe_values_ext,
    const dealii::FEValues<dim,dim> &/*fe_values_vol_int*/,
    const dealii::FEValues<dim,dim> &/*fe_values_vol_ext*/,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);

    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_int_ad(n_dofs_int);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_ext_ad(n_dofs_int);

    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);

    std::vector<ADArray> aux_soln_on_boundary_int(n_face_quad_pts);
    std::vector<ADArray> aux_soln_on_boundary_ext(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > aux_soln_int(n_face_quad_pts);
    std::vector< ADArrayTensor1 > aux_soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
    const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
        soln_coeff_int_ad[idof].diff(idof, n_total_indep);

        for (int idim =0; idim<dim; idim++){
            aux_soln_coeff_int_ad[idof][idim] = auxiliary_solution[idim][dof_indices_int[idof]];
        }
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
        soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);

        for (int idim =0; idim<dim; idim++){
            aux_soln_coeff_ext_ad[idof][idim] = auxiliary_solution[idim][dof_indices_ext[idof]];
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;

            aux_soln_int[iquad][istate]      = 0;
            aux_soln_ext[iquad][istate]      = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);

            aux_soln_int[iquad][istate]  += aux_soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);

            aux_soln_ext[iquad][istate]  += aux_soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
        }
        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        conv_phys_flux_int[iquad] = pde_physics->convective_flux (soln_int[iquad]);
        conv_phys_flux_ext[iquad] = pde_physics->convective_flux (soln_ext[iquad]);

        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = pde_physics->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
//get auxiliary solution on the boundary
        for (int s=0; s<nstate; s++) {
            aux_soln_on_boundary_int[iquad][s] = (aux_soln_int[iquad][s]) * normal_int;
            aux_soln_on_boundary_ext[iquad][s] = (aux_soln_ext[iquad][s]) * normal_ext;
        }
    }


    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux_int[iquad][istate]*normals_int[iquad];
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
           // rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) *  aux_soln_on_boundary_int[iquad][istate] *  JxW_int[iquad];
        }

        local_rhs_int_cell(itest_int) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR1_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR1_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_int, dR1_dW1);
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_ext, dR1_dW2);
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const ADtype flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_phys_flux_ext[iquad][istate]*(-normals_int[iquad]);
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
           // rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) *  aux_soln_on_boundary_ext[iquad][istate] *  JxW_int[iquad];
        }

        local_rhs_ext_cell(itest_ext) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR2_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR2_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_int, dR2_dW1);
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_ext, dR2_dW2);
        }
    }
}

/************************************************************************
 * Explicit
 * ***********************************************************************/

template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_volume_terms_explicit(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FEValues<dim,dim> &fe_values_vol_flux,
    const dealii::FEValues<dim,dim> &fe_values_vol_soln_flux,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    //std::cout << "assembling cell terms" << std::endl;
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;
    //FOR FLUX POINTS
    const unsigned int n_quad_pts_flux      = fe_values_vol_flux.n_quadrature_points;
    const unsigned int n_dofs_cell_flux     = fe_values_vol_flux.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    const std::vector<real> &JxW_soln_flux = fe_values_vol_soln_flux.get_JxW_values ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > aux_soln_at_q(n_quad_pts_flux);
    std::vector< ADArray > aux_soln_div_at_q(n_quad_pts_flux); // Tensor initialize with zeros

    std::vector< ADArray > source_at_q(n_quad_pts);

    //Evaluated at the flux points
    std::vector< ADArray > soln_at_q_flux(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > soln_grad_at_q_flux(n_quad_pts_flux); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts_flux);
    std::vector< ADArrayTensor1 > soln_at_q_split(n_quad_pts_flux);

    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
    //    soln_coeff[idof].diff(idof, n_dofs_cell);
        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff[idof][idim] = auxiliary_solution[idim][cell_dofs_indices[idof]];
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the flux volume quadrature points
            soln_at_q_flux[iquad][istate]      = 0;
            soln_grad_at_q_flux[iquad][istate] = 0;
            aux_soln_at_q[iquad][istate]  = 0;
        }
    }
    // Interpolate solution to Volume cubature nodes
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }

       // std::cout<< "Primary vol soln : "<< soln_at_q[iquad][0] <<std::endl;

        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
           // source_at_q[iquad] = pde_physics->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad]);
           // source_at_q[iquad] = pde_physics_double->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad]);
            source_at_q[iquad] = pde_physics_double->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad], DGBase<dim,real>::current_time);
        }
    }
   //flux evaluated at flux points In case different flux points for projection 
    for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q_flux[iquad][istate]      += soln_coeff[idof] * fe_values_vol_soln_flux.shape_value_component(idof, iquad, istate);
              soln_grad_at_q_flux[iquad][istate] += soln_coeff[idof] * fe_values_vol_soln_flux.shape_grad_component(idof, iquad, istate);

              aux_soln_at_q[iquad][istate]  += aux_soln_coeff[idof] * fe_values_vol_soln_flux.shape_value_component(idof, iquad, istate);
        }

       // std::cout<< "Primary vol soln aux: "<< aux_soln_at_q[iquad][0] <<std::endl;

        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term

       // conv_phys_flux_at_q[iquad] = pde_physics->convective_flux (soln_at_q_flux[iquad]);
        conv_phys_flux_at_q[iquad] = pde_physics_double->convective_flux (soln_at_q_flux[iquad]);//evaluate flux at flux node

    //    diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q_flux[iquad], soln_grad_at_q_flux[iquad]);
        for (int idim=0; idim<dim; idim++){
            for (int istate =0; istate<nstate; istate++){
                soln_at_q_split[iquad][istate][idim] = soln_at_q_flux[iquad][istate];
            }
        }

    }



//build projection for differentiation
//#if 0
    const unsigned int n_rows_dif_order_cells = DGBase<dim,real>::dif_order_cells.size() / 4;
    unsigned int idif_order_cell =0;
    for(; idif_order_cell<n_rows_dif_order_cells; idif_order_cell++){
        if(n_dofs_cell == DGBase<dim,real>::dif_order_cells[idif_order_cell][0]
            && n_quad_pts == DGBase<dim,real>::dif_order_cells[idif_order_cell][1])
            break;
    }

    dealii::FullMatrix<real> projection_matrix(n_dofs_cell_flux, n_quad_pts_flux);
    for(unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
        for( unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
            const unsigned int dof_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][2];
            const unsigned int quad_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][3];
            projection_matrix[idof][iquad] = DGBase<dim,real>::global_projection_operator[dof_flux_start + idof][quad_flux_start + iquad];//Projection operator
        }
    }
//#endif
//for skew symmteric form of metric terms
    std::vector< ADArrayTensor1 > conv_phys_flux_metric(n_quad_pts_flux);
    get_flux_with_metric_terms(fe_values_vol_flux, n_quad_pts_flux, n_dofs_cell, conv_phys_flux_at_q, conv_phys_flux_metric);//multiply metric terms with flux at the flux nodes


    //Get modal coefficients of the flux through projection operator

    std::vector< ADArrayTensor1 > conv_projected_phys_flux(n_dofs_cell_flux);
    std::vector< ADArrayTensor1 > conv_projected_phys_flux_metric(n_dofs_cell_flux);
    std::vector< ADArrayTensor1 > aux_soln_projected(n_dofs_cell_flux);
    std::vector< ADArrayTensor1 > soln_at_q_projected(n_dofs_cell_flux);
    for (unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
       // for (int istate=0; istate<nstate; istate++){
            const unsigned int istate = fe_values_vol_flux.get_fe().system_to_component_index(idof).first;
            conv_projected_phys_flux[idof][istate] = 0.0;
            conv_projected_phys_flux_metric[idof][istate] = 0.0;
            aux_soln_projected[idof][istate] = 0.0;
            soln_at_q_projected[idof][istate] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
                conv_projected_phys_flux[idof][istate] += projection_matrix[idof][iquad] * conv_phys_flux_at_q[iquad][istate];
                conv_projected_phys_flux_metric[idof][istate] += projection_matrix[idof][iquad] * conv_phys_flux_metric[iquad][istate];
                aux_soln_projected[idof][istate] += projection_matrix[idof][iquad] * aux_soln_at_q[iquad][istate];
                soln_at_q_projected[idof][istate] += projection_matrix[idof][iquad] * soln_at_q_split[iquad][istate];
            }
       // }
    } 


//get derivative of nonlinear flux
//note: fe_values.shape_grad is metric terms * derivative of basis function wrt ref. coord
//note: fe_collection_flux[].shape_grad is derivative of basis function wrt ref. coord
    std::vector<ADArray> flux_divergence(n_quad_pts_flux);
    std::vector<ADArray> flux_divergence_metric(n_quad_pts_flux);
    std::vector<ADArray> split_div(n_quad_pts_flux);//d(Chi)/d(xi)*u
    const unsigned int fe_index_curr_cell = pow(n_dofs_cell,1.0/dim) - 1;
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
            flux_divergence_metric[iquad][istate] = 0.0;
            aux_soln_div_at_q[iquad][istate] = 0.0;
            split_div[iquad][istate] = 0.0;
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].point(iquad);
            for (unsigned int idof=0; idof<n_dofs_cell_flux; ++idof) {
                //Metric^T * Df
                flux_divergence[iquad][istate] += conv_projected_phys_flux[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
                //D( Metric*flux)
                flux_divergence_metric[iquad][istate] += conv_projected_phys_flux_metric[idof][istate] * DGBase<dim,real>::fe_collection_flux[fe_index_curr_cell].shape_grad_component(idof, qpoint, istate);
                aux_soln_div_at_q[iquad][istate] += aux_soln_projected[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
                split_div[iquad][istate] += soln_at_q_projected[idof][istate] * fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
            }
        }
    }


//ESFR CLASSICAL 
//If do not want the (M+K)^{-1} on the volume term, multiply rhs by filter_chi = (M+K)M^{-1} * chi^T, if ESFR classical == False, filter_chi = chi^T

    dealii::FullMatrix<real> filter_chi(n_dofs_cell, n_quad_pts_flux);
    dealii::FullMatrix<real> filter_chi_source(n_dofs_cell);
    dealii::FullMatrix<real> K_operator(n_dofs_cell);
    std::vector<dealii::FullMatrix<real>> K_operator_aux(dim);
    for(int idim=0; idim<dim; idim++){
        K_operator_aux[idim].reinit(n_dofs_cell, n_dofs_cell);
    }
    DGBase<dim,real>::get_K_operator_FR(DGBase<dim,real>::fe_collection, fe_index_curr_cell, fe_values_vol, n_quad_pts, n_dofs_cell, fe_index_curr_cell, K_operator, K_operator_aux);
    get_Flux_Reconstruction_modifying_filters(fe_values_vol, n_dofs_cell, n_quad_pts, fe_values_vol_soln_flux, n_quad_pts_flux, K_operator, filter_chi, filter_chi_source);



//build derivative with projection
//
//FOR USING 2 point Flux split form commented out

#if 0
    std::vector<dealii::FullMatrix<real>> local_derivative_operator(dim);
    for(int idim=0; idim<dim; idim++){
       local_derivative_operator[idim].reinit(n_quad_pts_flux, n_dofs_cell_flux);
    }
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
       // const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].point(iquad);
        for (unsigned int idof=0; idof<n_dofs_cell_flux; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
            derivative = fe_values_vol_flux.shape_grad_component(idof, iquad, istate);
            //derivative = DGBase<dim,real>::fe_collection_flux[fe_index_curr_cell].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                local_derivative_operator[idim][iquad][idof] = derivative[idim];//store dChi/dXi
            }
        }
    }
    }
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> derivative_temp(n_quad_pts_flux, n_dofs_cell_flux);
        for(unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
                derivative_temp[iquad][idof] = local_derivative_operator[idim][iquad][idof];
            }
        }
       // Chi_inv_operator.mmult(local_derivative_operator[idim],derivative_temp);
       // projection_oper.mmult(local_derivative_operator[idim],derivative_temp);
        derivative_temp.mmult(local_derivative_operator[idim],projection_matrix);
    }

//Note split form will have to change since now in p+1 polynomial basis rather than nodal solutions of the flux
    // Evaluate flux divergence by interpolating the flux
    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
    //const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();
    //USING 2 Point Flux for Split form commented out
//#if 0 
    //std::vector<ADArrayTensor1 > flux_divergence_lagrange(n_quad_pts_flux);
    std::vector<ADArray> flux_divergence_lagrange(n_quad_pts_flux);
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
            flux_divergence_lagrange[iquad][istate] = 0.0;
            for ( unsigned int flux_basis = 0; flux_basis < n_quad_pts_flux; ++flux_basis ) {
                if (this->all_parameters->use_split_form == true)
                {
                  //  flux_divergence_lagrange[iquad][istate] += 2* pde_physics_double->convective_numerical_split_flux(soln_at_q_flux[iquad],soln_at_q_flux[flux_basis])[istate] *  fe_values_lagrange.shape_grad(flux_basis,iquad);
                    //flux_divergence_lagrange[iquad][istate] += 2* pde_physics_double->convective_numerical_split_flux(soln_at_q[iquad],soln_at_q[flux_basis])[istate] *  fe_values_vol.shape_grad(flux_basis,iquad);
                    //flux_divergence_lagrange[iquad][istate] += 2* pde_physics_double->convective_numerical_split_flux(soln_at_q_flux[iquad],soln_at_q_flux[flux_basis])[istate];
                    //#if 0
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence_lagrange[iquad][istate] += 2* pde_physics_double->convective_numerical_split_flux(soln_at_q_flux[iquad],soln_at_q_flux[flux_basis])[istate][idim] *  local_derivative_operator[idim][iquad][flux_basis];
                    }
                    //#endif
                }
                else
                {
               //     flux_divergence_lagrange[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad);
                //    #if 0
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence_lagrange[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate][idim] *  local_derivative_operator[idim][iquad][flux_basis];
                    }
                 //   #endif
                    //flux_divergence_lagrange[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate];
                }
            }
        }
    }
//#endif
#endif


    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0;
    //    ADtype split = 0;

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {

            // Convective
            // Now minus such 2 integrations by parts
    //        assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);

            double alpha = 1.0;
            double beta = 1.0;
        // #if 0
            if (this->all_parameters->use_split_form == true){
                alpha = 2.0/3.0;
                //for split form
                //split = split + (1.0 - alpha) * soln_at_q_flux[iquad][istate] *  fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * split_div[iquad][istate] * JxW_soln_flux[iquad];//THIS IS IT YAY
                split = split + (1.0 - alpha) * beta * soln_at_q_flux[iquad][istate] * filter_chi[itest][iquad]  * split_div[iquad][istate] * JxW_soln_flux[iquad];//THIS IS IT YAY
                
                //split = split + (1.0 - alpha) *  filter_chi[itest][iquad]  * split_div[iquad][istate] * JxW_soln_flux[iquad];//THIS IS IT YAY
            }
       // #endif
       // SKEW symmetric metric derivative
            if (this->all_parameters->use_skew_sym_deriv == true){
                beta = 1.0/2.0;
                const std::vector<real> &quad_weights = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].get_weights ();
                //rhs = rhs - (1.0 - beta) * alpha * filter_chi[itest][iquad] * flux_divergence_metric[iquad][istate] *  quad_weights[iquad];
                rhs = rhs - (1.0 - beta) * filter_chi[itest][iquad] * flux_divergence_metric[iquad][istate] *  quad_weights[iquad];
            }


            //rhs = rhs - alpha * fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];
            //rhs = rhs - alpha * beta * fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];
            //
            rhs = rhs - alpha * beta * filter_chi[itest][iquad] * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];
            //rhs = rhs - filter_chi[itest][iquad] * t_point[iquad][istate] * JxW_soln_flux[iquad];
      //      rhs = rhs - beta * filter_chi[itest][iquad] * flux_divergence_lagrange[iquad][istate] * JxW_soln_flux[iquad];
            //const std::vector<real> &quad_weights = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].get_weights ();
            //rhs = rhs - filter_chi[itest][iquad] * flux_divergence_lagrange[iquad][istate] * quad_weights[iquad];



            //rhs = rhs - alpha * filter_chi[itest][iquad] * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];
    //        rhs = rhs - alpha * fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * flux_divergence_metric[iquad][istate] * JxW_soln_flux[iquad];
//#endif
            //rhs = rhs - fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW_soln_flux[iquad];


            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
           // rhs = rhs + fe_values_vol_soln_flux.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW_soln_flux[iquad];
            rhs = rhs + fe_values_vol_soln_flux.shape_value_component(itest,iquad,istate) * aux_soln_div_at_q[iquad][istate] * JxW_soln_flux[iquad];
            // Source

            if(iquad < n_quad_pts){
                if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                   // rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
                    rhs = rhs +  filter_chi_source[itest][iquad]* source_at_q[iquad][istate] * JxW[iquad];
                }
            }
        }

#if 0
        if (this->all_parameters->use_split_form == true){
            //split = split * soln_at_q[itest][istate]; 
            //split = split * soln_coeff[itest];
            rhs = rhs - split;
        }
#endif

       // local_rhs_int_cell(itest) += rhs.val();
        local_rhs_int_cell(itest) += rhs;

        #if 0
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(cell_dofs_indices[itest], cell_dofs_indices, residual_derivatives);
        }
        #endif
    }
}

template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_boundary_term_explicit(
    const unsigned int boundary_id,
    const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> aux_soln_int(n_face_quad_pts);

    std::vector<ADArray> soln_ext(n_face_quad_pts);
    std::vector<ADArrayTensor1> aux_soln_ext(n_face_quad_pts);

    std::vector<ADArray> aux_soln_on_boundary(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_int(n_dofs_cell);
   // const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
//        soln_coeff_int[idof].diff(idof, n_total_indep);
        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff_int[idof][idim] = auxiliary_solution[idim][dof_indices_int[idof]];
        }
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            aux_soln_int[iquad][istate]      = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];
       // const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
            aux_soln_int[iquad][istate]  += aux_soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
        }
       // std::cout<< "Primary boundary soln : "<< soln_int[iquad][0] <<std::endl;
      //  std::cout<< "Primary boundary soln aux: "<< aux_soln_int[iquad][0] <<std::endl;

        const dealii::Point<dim, real> x_quad = quad_pts[iquad];
        pde_physics_double->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.

       // conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        // Used for strong form
        // Which physical convective flux to use?
       // conv_phys_flux[iquad] = pde_physics->convective_flux (soln_int[iquad]);
        conv_phys_flux[iquad] = pde_physics_double->convective_flux (soln_int[iquad]);

    #if 0
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    #endif
       // diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);

        for (int s=0; s<nstate; s++) {
            aux_soln_on_boundary[iquad][s] = (aux_soln_int[iquad][s]) * normal_int;
        }
    }

    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
           // rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) *  aux_soln_on_boundary[iquad][istate] *  JxW[iquad];
        }
        // *******************

       // local_rhs_int_cell(itest) += rhs.val();
        local_rhs_int_cell(itest) += rhs;

        #if 0
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(dof_indices_int[itest], dof_indices_int, residual_derivatives);
        }
        #endif
    }
}

template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_face_term_explicit(
    const dealii::FEValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValues<dim,dim>     &fe_values_ext,
    const dealii::FEValues<dim,dim> &fe_values_vol_int,
    const dealii::FEValues<dim,dim> &fe_values_vol_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    //std::cout << "assembling face terms" << std::endl;
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,real>;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);

    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_int_ad(n_dofs_int);
    std::vector<dealii::Tensor<1,dim,real>> aux_soln_coeff_ext_ad(n_dofs_int);

    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_face_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_face_flux_ext(n_face_quad_pts);

    std::vector<ADArray> aux_soln_on_boundary_int(n_face_quad_pts);
    std::vector<ADArray> aux_soln_on_boundary_ext(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > aux_soln_int(n_face_quad_pts);
    std::vector< ADArrayTensor1 > aux_soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext

    // AD variable
   // const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
       // soln_coeff_int_ad[idof].diff(idof, n_total_indep);

        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff_int_ad[idof][idim] = auxiliary_solution[idim][dof_indices_int[idof]];
        }
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
       // soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);

        for (int idim=0; idim<dim; idim++){
            aux_soln_coeff_ext_ad[idof][idim] = auxiliary_solution[idim][dof_indices_ext[idof]];
        }
    }
//get modal coefficients of flux to interpolate to face
    const unsigned int n_quad_pts_int = fe_values_vol_int.n_quadrature_points;
    const unsigned int n_quad_pts_ext = fe_values_vol_ext.n_quadrature_points;
    std::vector< ADArray > soln_at_q_int(n_quad_pts_int);
    std::vector< ADArray > soln_at_q_ext(n_quad_pts_ext);
    for (unsigned int iquad=0; iquad<n_quad_pts_int; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_at_q_int[iquad][istate]      = 0.0;
            soln_at_q_ext[iquad][istate]      = 0.0;
        }
    }
    std::vector< ADArrayTensor1 > convective_phys_flux_at_q_int(n_quad_pts_int);
    std::vector< ADArrayTensor1 > convective_phys_flux_at_q_ext(n_quad_pts_ext);
    for(unsigned int iquad=0; iquad<n_quad_pts_int; iquad++){
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_vol_int.get_fe().system_to_component_index(idof).first;
              soln_at_q_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_vol_int.shape_value_component(idof, iquad, istate);
        }
        convective_phys_flux_at_q_int[iquad] = pde_physics_double->convective_flux (soln_at_q_int[iquad]);
    }
    for(unsigned int iquad=0; iquad<n_quad_pts_ext; iquad++){
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_vol_ext.get_fe().system_to_component_index(idof).first;
              soln_at_q_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_vol_ext.shape_value_component(idof, iquad, istate);
        }
        convective_phys_flux_at_q_ext[iquad] = pde_physics_double->convective_flux (soln_at_q_ext[iquad]);
    }
    const unsigned int n_rows_dif_order_cells = DGBase<dim,real>::dif_order_cells.size() / 4;
    unsigned int idif_order_cell =0;
    for(; idif_order_cell<n_rows_dif_order_cells; idif_order_cell++){
        if(n_dofs_int == DGBase<dim,real>::dif_order_cells[idif_order_cell][0]
            && n_quad_pts_int == DGBase<dim,real>::dif_order_cells[idif_order_cell][1])
            break;
    }

    dealii::FullMatrix<real> projection_matrix(n_dofs_int, n_quad_pts_int);
    for(unsigned int idof=0; idof<n_dofs_int; idof++){
        for( unsigned int iquad=0; iquad<n_quad_pts_int; iquad++){
            const unsigned int dof_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][2];
            const unsigned int quad_flux_start = DGBase<dim,real>::dif_order_cells[idif_order_cell][3];
            projection_matrix[idof][iquad] = DGBase<dim,real>::global_projection_operator[dof_flux_start + idof][quad_flux_start + iquad];
        }
    }
    std::vector< ADArrayTensor1 > conv_projected_phys_flux_int(n_dofs_int);
    std::vector< ADArrayTensor1 > conv_projected_phys_flux_ext(n_dofs_ext);
    for (unsigned int idof=0; idof<n_dofs_int; idof++){
            const unsigned int istate = fe_values_vol_int.get_fe().system_to_component_index(idof).first;
            conv_projected_phys_flux_int[idof][istate] = 0.0;
            conv_projected_phys_flux_ext[idof][istate] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts_int; iquad++){
                conv_projected_phys_flux_int[idof][istate] += projection_matrix[idof][iquad] * convective_phys_flux_at_q_int[iquad][istate];
                conv_projected_phys_flux_ext[idof][istate] += projection_matrix[idof][iquad] * convective_phys_flux_at_q_ext[iquad][istate];
            }
    } 

//end of get modal coeff of flux

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0.0;
            soln_grad_int[iquad][istate] = 0.0;
            soln_ext[iquad][istate]      = 0.0;
            soln_grad_ext[iquad][istate] = 0.0;

            aux_soln_int[iquad][istate]      = 0.0;
            aux_soln_ext[iquad][istate]      = 0.0;

            conv_phys_flux_int[iquad][istate]      = 0.0;
            conv_phys_flux_ext[iquad][istate]      = 0.0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);

            aux_soln_int[iquad][istate]  += aux_soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);


            conv_phys_flux_int[iquad][istate] += conv_projected_phys_flux_int[idof][istate] * fe_values_int.shape_value_component(idof, iquad, istate);
        }
       // std::cout<< "Primary face soln : "<< soln_int[iquad][0] <<std::endl;
       // std::cout<< "Primary face soln aux : "<< aux_soln_int[iquad][0] <<std::endl;
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);

            aux_soln_ext[iquad][istate]  += aux_soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);


            conv_phys_flux_ext[iquad][istate] += conv_projected_phys_flux_ext[idof][istate] * fe_values_ext.shape_value_component(idof, iquad, istate);
        }
        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term

        //std::cout <<"evaluating numerical fluxes" <<std::endl;
       // conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

#if 0
   // std::array<dealii::Tensor<1,dim,real>,nstate> flux_avg;
   // for (int s=0; s<nstate; s++) {
   //     flux_avg[s] = 0.5*(conv_phys_flux_int[iquad][s] + conv_phys_flux_ext[iquad][s]);
   // }
    std::array<real,nstate> flux_avg;
    for (int s=0; s<nstate; s++) {
       // flux_avg[s] = 0.25*(soln_int[iquad][s]*soln_int[iquad][s] + soln_ext[iquad][s]*soln_ext[iquad][s]);
        flux_avg[s] = 0.5*(conv_phys_flux_int[iquad][s] + conv_phys_flux_ext[iquad][s])*normal_int;
    }
   // const real conv_max_eig_int = pde_physics_double->max_convective_eigenvalue(soln_int[iquad]);
   // const real conv_max_eig_ext = pde_physics_double->max_convective_eigenvalue(soln_ext[iquad]);
   // const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
    double lambda; 
    if(std::abs(soln_int[iquad][0]-soln_ext[iquad][0])<1e-12){
        lambda=0.0;
    }
    else{
        lambda =1.0/6.0*soln_int[iquad][0]*conv_phys_flux_int[iquad][0]*normal_int - 1.0/6.0*soln_ext[iquad][0]*conv_phys_flux_ext[iquad][0]*normal_int +1.0/4.0*( -pow(soln_int[iquad][0],3)-pow(soln_ext[iquad][0],2)*soln_int[iquad][0]+pow(soln_ext[iquad][0],3)+pow(soln_int[iquad][0],2)*soln_ext[iquad][0]);
        lambda /=(soln_ext[iquad][0]-soln_int[iquad][0]);
    }
   // lambda= 1.0/12.0 *pow(soln_ext[iquad][0]-soln_int[iquad][0],2);
    lambda= 1.0/2.0 *std::max(std::abs(soln_int[iquad][0]),std::abs(soln_ext[iquad][0])) * (soln_ext[iquad][0]-soln_int[iquad][0]);
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
       // numerical_flux_dot_n[s] = flux_avg[s]*normal_int - conv_max_eig * (soln_ext[iquad][s]-soln_int[iquad][s]);
        numerical_flux_dot_n[s] = flux_avg[s] - lambda;
    }
        conv_num_flux_dot_n[iquad] = numerical_flux_dot_n;
#endif




       // conv_phys_flux_int[iquad] = pde_physics->convective_flux (soln_int[iquad]);
     //   conv_phys_flux_int[iquad] = pde_physics_double->convective_flux (soln_int[iquad]);
        conv_face_flux_int[iquad] = pde_physics_double->convective_flux (soln_int[iquad]);
       // conv_phys_flux_ext[iquad] = pde_physics->convective_flux (soln_ext[iquad]);
      //  conv_phys_flux_ext[iquad] = pde_physics_double->convective_flux (soln_ext[iquad]);
        conv_face_flux_ext[iquad] = pde_physics_double->convective_flux (soln_ext[iquad]);

       // std::cout <<"done evaluating numerical fluxes" <<std::endl;


       // diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext;
        }
       // diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
       // diss_flux_jump_ext[iquad] = pde_physics->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);
        diss_flux_jump_int[iquad] = pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

       // diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
//get auxiliary solution on the boundary
        for (int s=0; s<nstate; s++) {
            aux_soln_on_boundary_int[iquad][s] = (aux_soln_int[iquad][s]) * normal_int;
            aux_soln_on_boundary_ext[iquad][s] = (aux_soln_ext[iquad][s]) * normal_ext;
        }
    }



//Chi^T*P^T*Chi_face
#if 0
    const unsigned int fe_index_curr_cell = ndofs_cell_int - 1;
    dealii::FullMatrix<real> Chi_operator(n_dofs_cell_int +1, n_dofs_cell_int);
    dealii::FullMatrix<real> Phi_operator(n_dofs_cell_int + 1, n_dofs_cell_int + 1);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell_int; ++itest) {
        for (unsigned int iquad=0; iquad<n_dofs_cell_int +1; ++iquad) {
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection_flux[fe_index_curr_cell].point(iquad);
            Chi_operator[iquad][itest] = DGBase<dim,real>::fe_collection[fe_index_curr_cell].shape_value_component(itest,qpoint,istate);
        }
    }
    }
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell_int +1; ++itest) {
        for (unsigned int iquad=0; iquad<n_dofs_cell_int+1; ++iquad) {
            const dealii::Point<dim> qpoint  = DGBase<dim,real>::volume_quadrature_collection[fe_index_curr_cell].point(iquad);
            Phi_operator[iquad][itest] = DGBase<dim,real>::fe_collection_flux[fe_index_curr_cell].shape_value_component(itest,qpoint,istate);
        }
    }
    }
    dealii::FullMatrix<real> Phi_operator_inv(n_dofs_cell_int + 1, n_dofs_cell_int + 1);
    dealii::FullMatrix<real> proj(n_dofs_cell_int, n_dofs_cell_int + 1);
    dealii::FullMatrix<real> proj_face(n_dofs_cell_int, n_face_quad_pts);
    Phi_operator_inv.invert(Phi_operator);
    Chi_operator.TmTmult(proj, Phi_operator);
    for(unsigned int idof=0; idof<n_dofs_cell_int; idof++){
        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            proj_face[idof][iquad]; 
        }
    }
#endif

  //  std::cout <<"normal "<<normals_int[0]<<std::endl;


    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            double alpha= 1.0;
            if (this->all_parameters->use_split_form == true){
                alpha = 2.0/3.0;
            }
            const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - (alpha *conv_phys_flux_int[iquad][istate] +(1.0-alpha)*conv_face_flux_int[iquad][istate] )*normals_int[iquad];
            
           // const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux_int[iquad][istate]*normals_int[iquad];
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
           // rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
           //
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) *  aux_soln_on_boundary_int[iquad][istate] *  JxW_int[iquad];
        }

       // local_rhs_int_cell(itest_int) += rhs.val();
        local_rhs_int_cell(itest_int) += rhs;
        #if 0
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR1_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR1_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_int, dR1_dW1);
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_ext, dR1_dW2);
        }
        #endif
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            double alpha= 1.0;
            if (this->all_parameters->use_split_form == true){
                alpha = 2.0/3.0;
            }
            const ADtype flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - (alpha*conv_phys_flux_ext[iquad][istate]+(1.0-alpha)*conv_face_flux_ext[iquad][istate])*(-normals_int[iquad]);
           // const ADtype flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_phys_flux_ext[iquad][istate]*(-normals_int[iquad]);
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
           // rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
           //
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) *  aux_soln_on_boundary_ext[iquad][istate] *  JxW_int[iquad];
        }

       // local_rhs_ext_cell(itest_ext) += rhs.val();
        local_rhs_ext_cell(itest_ext) += rhs;
        #if 0
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR2_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR2_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_int, dR2_dW1);
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_ext, dR2_dW2);
        }
        #endif
    }
}
/**********************************************************************************
 * MAIN for DG Strong
 * *********************************************************************************/

template <int dim, int nstate, typename real>
void DGStrong<dim,nstate,real>::assemble_residual (const bool compute_dRdW)
{
    DGBase<dim,real>::right_hand_side = 0;

    DGBase<dim,real>::build_global_projection_operator();

    if (compute_dRdW) DGBase<dim,real>::system_matrix = 0;

    // For now assume same polynomial degree across domain
    const unsigned int max_dofs_per_cell = DGBase<dim,real>::dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);

    const auto mapping = (*(DGBase<dim,real>::high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_neigh (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    //dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of interior face.
    //dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection_flux, this->neighbor_face_update_flags); ///< FEValues of exterior face.

    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, DGBase<dim,real>::fe_collection_lagrange, DGBase<dim,real>::volume_quadrature_collection, this->volume_update_flags);

//#if 0
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of subface.
//#endif

//over integrate
#if 0
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection_flux, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of subface.
#endif

//solution basis functions evaluated at flux volume nodes
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_soln_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.

    unsigned int n_cell_visited = 0;
    unsigned int n_face_visited = 0;

    DGBase<dim,real>::solution.update_ghost_values();

    //LOOP FOR AUXILIARY EQUATION


    //initialize the rhs for aux to be stored
    DGBase<dim,real>::locally_owned_dofs = DGBase<dim,real>::dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(DGBase<dim,real>::dof_handler, DGBase<dim,real>::ghost_dofs);
    DGBase<dim,real>::locally_relevant_dofs = DGBase<dim,real>::ghost_dofs;
    DGBase<dim,real>::ghost_dofs.subtract_set(DGBase<dim,real>::locally_owned_dofs);
    auxiliary_RHS.resize(dim);
    auxiliary_solution.resize(dim);
    for (int idim=0; idim<dim; idim++){
        auxiliary_RHS[idim].reinit(DGBase<dim,real>::locally_owned_dofs, DGBase<dim,real>::ghost_dofs, mpi_communicator);
        auxiliary_solution[idim].reinit(DGBase<dim,real>::locally_owned_dofs, DGBase<dim,real>::ghost_dofs, mpi_communicator);
    }

    for (auto current_cell = DGBase<dim,real>::dof_handler.begin_active(); current_cell != DGBase<dim,real>::dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = DGBase<dim,real>::fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for (unsigned int i=0; i<n_dofs_curr_cell ; ++i) {
            for (int idim=0; idim<dim; idim++){
                auxiliary_RHS[idim][current_dofs_indices[i]] = 0.0;
                auxiliary_solution[idim][current_dofs_indices[i]] = 0.0;
            }
        }
    }

    if (DGBase<dim,real>::all_parameters->pde_type == 1 || DGBase<dim,real>::all_parameters->pde_type == 2)//1 is diffusion, 2 is convection-diffusion
    {

    for (auto current_cell = DGBase<dim,real>::dof_handler.begin_active(); current_cell != DGBase<dim,real>::dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        n_cell_visited++;


        // Current reference element related to this physical cell
        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;
        const dealii::FESystem<dim,dim> &current_fe_ref = DGBase<dim,real>::fe_collection[fe_index_curr_cell];
       // const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Local vector contribution from each cell
        std::vector<dealii::Tensor<1,dim,double>> current_cell_rhs_aux (n_dofs_curr_cell ); // Defaults to 0.0 initialization

        //initialize current cell's rhs for aux eqn to zero
        for (unsigned int idof = 0 ; idof<n_dofs_curr_cell; idof++)
        {
            for (int idim =0; idim < dim; idim++)
            {  
                current_cell_rhs_aux[idof][idim] = 0.0;
            }
        }

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);

        //FOR FLUX POINTS
        const dealii::FESystem<dim,dim> &current_fe_ref_flux = DGBase<dim,real>::fe_collection_flux[fe_index_curr_cell];


        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        if ( compute_dRdW ) {
            assemble_volume_terms_auxiliary_equation(fe_values_volume, current_dofs_indices, current_cell_rhs_aux);
        } else {
            assemble_volume_terms_auxiliary_equation(fe_values_volume, current_dofs_indices, current_cell_rhs_aux);
        }

        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

            auto current_face = current_cell->face(iface);
            auto neighbor_cell = current_cell->neighbor(iface);

            // See tutorial step-30 for breakdown of 4 face cases

            // Case 1:
            // Face at boundary
            if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {

                n_face_visited++;

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);

                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
                {
                    int cell_index  = current_cell->index();
                    //int cell_index = current_cell->index();
                    if (cell_index == 0 && iface == 0)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = DGBase<dim,real>::dof_handler.begin_active();
                        for (unsigned int i = 0 ; i < DGBase<dim,real>::triangulation->n_active_cells() - 1; ++i)
                        {
                            ++neighbor_cell;
                        }
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                         const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;

                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1,quad_index_neigh_cell,mapping_index_neigh_cell,fe_index_neigh_cell);

                    }
                    else if (cell_index == (int) DGBase<dim,real>::triangulation->n_active_cells() - 1 && iface == 1)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = DGBase<dim,real>::dof_handler.begin_active();
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;
                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
                    }

                    //std::cout << "cell " << current_cell->index() << "'s " << iface << "th face has neighbour: " << neighbor_cell->index() << std::endl;
                    //const int neighbor_face_no = (iface ==1) ? 0:1;
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();

                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                   // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();


                    std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
                    for (unsigned int idof = 0 ; idof<n_dofs_neigh_cell; idof++)
                    {
                        for (int idim =0; idim < dim; idim++)
                        {  
                            neighbor_cell_rhs[idof][idim] = 0.0;
                        }
                    }


                    if ( compute_dRdW ) {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    }

                } else {
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

                    const unsigned int boundary_id = current_face->boundary_id();
                    // Need to somehow get boundary type from the mesh
                    if ( compute_dRdW ) {
                    assemble_boundary_term_auxiliary_equation (boundary_id, fe_values_face_int, current_dofs_indices, current_cell_rhs_aux);
                    } else {
                    assemble_boundary_term_auxiliary_equation (boundary_id, fe_values_face_int, current_dofs_indices, current_cell_rhs_aux);
                    }
                }

                //CASE 1.5: periodic boundary conditions
                //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
            } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){

                neighbor_cell = current_cell->periodic_neighbor(iface);
                //std::cout << "cell " << current_cell->index() << " at boundary" <<std::endl;
                //std::cout << "periodic neighbour on face " << iface << " is " << neighbor_cell->index() << std::endl;


                if (!current_cell->periodic_neighbor_is_coarser(iface) &&
                    (neighbor_cell->index() > current_cell->index() ||
                     (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                   )
                {
                     n_face_visited++;
                    Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());


                    // Corresponding face of the neighbor.
                    // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                    const unsigned int neighbor_face_no = current_cell->periodic_neighbor_of_periodic_neighbor(iface);

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                   // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    // Local rhs contribution from neighbor
                    std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
                    for (unsigned int idof = 0 ; idof<n_dofs_neigh_cell; idof++)
                    {
                        for (int idim =0; idim < dim; idim++)
                        {  
                            neighbor_cell_rhs[idof][idim] = 0.0;
                        }
                    }

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();


                    if ( compute_dRdW ) {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    }

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        for (int idim=0; idim<dim; idim++){
                            auxiliary_RHS[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs[i][idim];
                        }
                    }
                }
                else
                {
                    //do nothing
                }


            // Case 2:
            // Neighbour is finer occurs if the face has children
            // In this case, we loop over the current large face's subfaces and visit multiple neighbors
            } else if (current_face->has_children()) {

                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                // Obtain cell neighbour
                const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);

                for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {

                    n_face_visited++;

                    // Get neighbor on ith subface
                    auto neighbor_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
                    // Since the neighbor cell is finer than the current cell, it should not have more children
                    Assert (!neighbor_cell->has_children(), dealii::ExcInternalError());

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                   // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
                    for (unsigned int idof = 0 ; idof<n_dofs_neigh_cell; idof++)
                    {
                        for (int idim =0; idim < dim; idim++)
                        {  
                            neighbor_cell_rhs[idof][idim] = 0.0;
                        }
                    }

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_subface.reinit (current_cell, iface, subface_no, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();

                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    if ( compute_dRdW ) {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_auxiliary(
                                fe_values_face_int, fe_values_face_ext,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs_aux, neighbor_cell_rhs);
                    }
                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        for (int idim=0; idim<dim; idim++){
                            auxiliary_RHS[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs[i][idim];
                        }
                    }
                }

            // Case 3:
            // Neighbor cell is NOT coarser
            // Therefore, they have the same coarseness, and we need to choose one of them to do the work
            } else if (
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
                    // We cannot use the cell->index() because the index is relative to the distributed triangulation
                    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
                 && neighbor_cell->is_ghost()
                 && current_cell->subdomain_id() < neighbor_cell->subdomain_id()
                )
                ||
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a local cell, we let the cell with the lower index do the work on that face
                 && neighbor_cell->is_locally_owned()
                 &&
                    (  // Cell with lower index does work
                       current_cell->index() < neighbor_cell->index()
                     ||
                       // If both cells have same index
                       // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
                       // then cell at the lower level does the work
                       (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                )
            )
            {
                n_face_visited++;
                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
                // Corresponding face of the neighbor.
                // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);

                // Get information about neighbor cell
                const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                const unsigned int mapping_index_neigh_cell = 0;
                const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
               // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                // Local rhs contribution from neighbor
                    std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
                    for (unsigned int idof = 0 ; idof<n_dofs_neigh_cell; idof++)
                    {
                        for (int idim =0; idim < dim; idim++)
                        {  
                            neighbor_cell_rhs[idof][idim] = 0.0;
                        }
                    }

                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();


                if ( compute_dRdW ) {
                        assemble_face_term_auxiliary(
                            fe_values_face_int, fe_values_face_ext,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs_aux, neighbor_cell_rhs);
                } else {
                        assemble_face_term_auxiliary(
                            fe_values_face_int, fe_values_face_ext,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs_aux, neighbor_cell_rhs);
                }

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    for (int idim=0; idim<dim; idim++){
                        auxiliary_RHS[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs[i][idim];
                    }
                }
            } else {
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces
            }

        } // end of face loop

        for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
            for (int idim=0; idim<dim; idim++){
                auxiliary_RHS[idim][current_dofs_indices[i]] += current_cell_rhs_aux[i][idim];
            }
        }

    } // end of cell loop

//q = inv(M) * current_cell_rhs_aux
//done for each dimencion of auxiliary variable

    //unsigned int n_dofs = DGBase<dim,real>::dof_handler.n_dofs();
   // DGBase<dim,real>::evaluate_mass_matrices(true);
    for(int idim =0; idim<dim; idim++){
       // DGBase<dim,real>::global_inverse_mass_matrix.vmult(auxiliary_solution[idim], auxiliary_RHS[idim]);
        DGBase<dim,real>::global_inverse_mass_correction_matrix[idim].vmult(auxiliary_solution[idim], auxiliary_RHS[idim]);
    }

    n_cell_visited = 0;
    n_face_visited = 0;

    for (int idim=0; idim<dim; idim++){
        auxiliary_solution[idim].update_ghost_values();
    }

    }//end of has diffusion


    //LOOP for Primary Equation
    for (auto current_cell = DGBase<dim,real>::dof_handler.begin_active(); current_cell != DGBase<dim,real>::dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        n_cell_visited++;


        // Current reference element related to this physical cell
        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;
        const dealii::FESystem<dim,dim> &current_fe_ref = DGBase<dim,real>::fe_collection[fe_index_curr_cell];
        const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Local vector contribution from each cell
        dealii::Vector<double> current_cell_rhs (n_dofs_curr_cell); // Defaults to 0.0 initialization

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);

        //FOR FLUX POINTS
        const dealii::FESystem<dim,dim> &current_fe_ref_flux = DGBase<dim,real>::fe_collection_flux[fe_index_curr_cell];


        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();


        dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);

        fe_values_collection_volume_lagrange.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_lagrange = fe_values_collection_volume_lagrange.get_present_fe_values();

        fe_values_collection_volume_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

        const dealii::FEValues<dim,dim> &fe_values_volume_flux = fe_values_collection_volume_flux.get_present_fe_values();

        fe_values_collection_volume_soln_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

        const dealii::FEValues<dim,dim> &fe_values_volume_soln_flux = fe_values_collection_volume_soln_flux.get_present_fe_values();


        if ( compute_dRdW ) {
            assemble_volume_terms_implicit (fe_values_volume, fe_values_volume_flux, fe_values_volume_soln_flux, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
        } else {
            assemble_volume_terms_explicit (fe_values_volume, fe_values_volume_flux, fe_values_volume_soln_flux, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
        }

        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

            auto current_face = current_cell->face(iface);
            auto neighbor_cell = current_cell->neighbor(iface);

            // See tutorial step-30 for breakdown of 4 face cases

            // Case 1:
            // Face at boundary
            if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {

                n_face_visited++;

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);

                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
                {
//#if 0
                    const unsigned int boundary_id_check = current_face->boundary_id();//not to double count periodic faces
                    if(boundary_id_check == 0)
                        continue;
                 //   printf(" Boundary %d\n",current_face->boundary_id());
                  //  fflush(stdout);
                    int cell_index  = current_cell->index();
                    //int cell_index = current_cell->index();
                    int neighbor_face = 0;
                    neighbor_cell = DGBase<dim,real>::dof_handler.begin_active();
                    fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                    int flag = 0;
                    for (auto neighbor_cell2 = DGBase<dim,real>::dof_handler.begin_active(); neighbor_cell2 != DGBase<dim,real>::dof_handler.end(); ++neighbor_cell2) {
                        if (!neighbor_cell2->is_locally_owned()) continue;
                        int neighbor_cell_index = neighbor_cell2->index();
                        if (neighbor_cell_index == cell_index)
                            continue;
                        for(int face_neigh=0; face_neigh<2; face_neigh++){
                            auto neigh_face = neighbor_cell2->face(face_neigh);
                            if(neigh_face->at_boundary()){
                                neighbor_cell = neighbor_cell2;
                                neighbor_face = face_neigh; 
                                flag = 1;
                                break; 
                            }
                        }
                        if(flag == 1)
                            break;
                    }
                    if(flag == 0){
                        printf("Couldn't Find periodic Neighbour\n");
                        fflush(stdout);
                    }
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;
                        fe_values_collection_face_ext.reinit(neighbor_cell, neighbor_face, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
                    const int neighbor_face_no = neighbor_face;
       // printf(" boundary %d int face %d ext face %d  current cell %d neighbour cell %d \n",boundary_id_check, iface, neighbor_face,cell_index,neighbor_cell->index());
//#endif
                
#if 0
                    int cell_index  = current_cell->index();
                    if (cell_index == 0 && iface == 0)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = DGBase<dim,real>::dof_handler.begin_active();
                        for (unsigned int i = 0 ; i < DGBase<dim,real>::triangulation->n_active_cells() - 1; ++i)
                        {
                            ++neighbor_cell;
                        }
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                         const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;

                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1,quad_index_neigh_cell,mapping_index_neigh_cell,fe_index_neigh_cell);

                    }
                    else if (cell_index == (int) DGBase<dim,real>::triangulation->n_active_cells() - 1 && iface == 1)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = DGBase<dim,real>::dof_handler.begin_active();
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;
                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
                    }
                if (
                    (neighbor_cell->index() < current_cell->index() ||
                     (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                   )
#endif
                    {//start of check for 1D periodic neighbour
#if 0

                    //std::cout << "cell " << current_cell->index() << "'s " << iface << "th face has neighbour: " << neighbor_cell->index() << std::endl;
                    const int neighbor_face_no = (iface ==1) ? 0:1;
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();

        //            const unsigned int boundary_id_check = current_face->boundary_id();//not to double count periodic faces
       // printf(" boundary %d int face %d ext face %d \n",boundary_id_check, iface, neighbor_face_no);

#endif

                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization


                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;

                    real penalty = 0.5 * ( penalty1 + penalty2 );
    

                    //#if 0
                    //for test case
                    penalty = DGBase<dim,real>::penalty;
                    //#endif
                    fe_values_collection_volume_neigh.reinit (neighbor_cell, quad_index, mapping_index, fe_index_neigh_cell);
                    const dealii::FEValues<dim,dim> &fe_values_volume_neigh = fe_values_collection_volume_neigh.get_present_fe_values();


                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                                    fe_values_face_int, fe_values_face_ext,
                                                    fe_values_volume, fe_values_volume_neigh,
                                                    penalty,
                                                    current_dofs_indices, neighbor_dofs_indices,
                                                    current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                                                    fe_values_face_int, fe_values_face_ext,
                                                    fe_values_volume, fe_values_volume_neigh,
                                                    penalty,
                                                    current_dofs_indices, neighbor_dofs_indices,
                                                    current_cell_rhs, neighbor_cell_rhs);
                    }
                        // Add local contribution from neighbor cell to global vector
                        for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                            DGBase<dim,real>::right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                        }
                    }//end check for 1D periodic

                } else {
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int normal_direction = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction);

                    real penalty = deg1sq / vol_div_facearea1;
                   // #if 0
                    //for test case
                    //penalty = this->penalty;
                    penalty = DGBase<dim,real>::penalty;
                   // #endif

                    const unsigned int boundary_id = current_face->boundary_id();
                    // Need to somehow get boundary type from the mesh
                    if ( compute_dRdW ) {
                        assemble_boundary_term_implicit (boundary_id, fe_values_face_int, fe_values_volume, penalty, current_dofs_indices, current_cell_rhs);
                    } else {
                        assemble_boundary_term_explicit (boundary_id, fe_values_face_int, fe_values_volume, penalty, current_dofs_indices, current_cell_rhs);
                    }
                }

                //CASE 1.5: periodic boundary conditions
                //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
            } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){

                neighbor_cell = current_cell->periodic_neighbor(iface);
                //std::cout << "cell " << current_cell->index() << " at boundary" <<std::endl;
                //std::cout << "periodic neighbour on face " << iface << " is " << neighbor_cell->index() << std::endl;

#if 0
                const unsigned int boundary_id_check = current_face->boundary_id();
                if(boundary_id_check % 2 == 0)//not to double count periodic face
                    continue;

                if (!current_cell->periodic_neighbor_is_coarser(iface)) 
#endif
//#if 0
                if (!current_cell->periodic_neighbor_is_coarser(iface) &&
                    (neighbor_cell->index() > current_cell->index() ||
                     (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                   )
//#endif
                {
                     n_face_visited++;
                    Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());


                    // Corresponding face of the neighbor.
                    // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                    const unsigned int neighbor_face_no = current_cell->periodic_neighbor_of_periodic_neighbor(iface);

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    // Local rhs contribution from neighbor
                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;

                    real penalty = 0.5 * ( penalty1 + penalty2 );
//printf("pen %g\n",penalty);
//fflush(stdout);
                    //penalty = 1;//99;
                   // #if 0
                    //for test case
                   // penalty = this->penalty;
                  // #if 0
                    penalty = DGBase<dim,real>::penalty;
                   // #endif
                   // #endif
//printf("pen %g\n",penalty);
//fflush(stdout);
                    fe_values_collection_volume_neigh.reinit (neighbor_cell, quad_index, mapping_index, fe_index_neigh_cell);
                    const dealii::FEValues<dim,dim> &fe_values_volume_neigh = fe_values_collection_volume_neigh.get_present_fe_values();

                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                fe_values_face_int, fe_values_face_ext,
                                fe_values_volume, fe_values_volume_neigh,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                                fe_values_face_int, fe_values_face_ext,
                                fe_values_volume, fe_values_volume_neigh,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    }

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        DGBase<dim,real>::right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                    }
                }
                else
                {
                    //do nothing
                }


            // Case 2:
            // Neighbour is finer occurs if the face has children
            // In this case, we loop over the current large face's subfaces and visit multiple neighbors
            } else if (current_face->has_children()) {

                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                // Obtain cell neighbour
                const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);

                for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {

                    n_face_visited++;

                    // Get neighbor on ith subface
                    auto neighbor_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
                    // Since the neighbor cell is finer than the current cell, it should not have more children
                    Assert (!neighbor_cell->has_children(), dealii::ExcInternalError());

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_subface.reinit (current_cell, iface, subface_no, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();

                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;
                    
                    real penalty = 0.5 * ( penalty1 + penalty2 );
                   // #if 0
                    //for test case
                   // penalty = this->penalty;
                    penalty = DGBase<dim,real>::penalty;
                   // #endif
                    fe_values_collection_volume_neigh.reinit (neighbor_cell, quad_index, mapping_index, fe_index_neigh_cell);
                    const dealii::FEValues<dim,dim> &fe_values_volume_neigh = fe_values_collection_volume_neigh.get_present_fe_values();

                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                fe_values_face_int, fe_values_face_ext,
                                fe_values_volume, fe_values_volume_neigh,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                            fe_values_face_int, fe_values_face_ext,
                            fe_values_volume, fe_values_volume_neigh,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                    }
                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        DGBase<dim,real>::right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                    }
                }

            // Case 3:
            // Neighbor cell is NOT coarser
            // Therefore, they have the same coarseness, and we need to choose one of them to do the work
            } else if (
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
                    // We cannot use the cell->index() because the index is relative to the distributed triangulation
                    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
                 && neighbor_cell->is_ghost()
                 && current_cell->subdomain_id() < neighbor_cell->subdomain_id()
                )
                ||
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a local cell, we let the cell with the lower index do the work on that face
                 && neighbor_cell->is_locally_owned()
                 &&
                    (  // Cell with lower index does work
                       current_cell->index() < neighbor_cell->index()
                     ||
                       // If both cells have same index
                       // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
                       // then cell at the lower level does the work
                       (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                )
            )
            {
                n_face_visited++;
                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
                // Corresponding face of the neighbor.
                // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);

                // Get information about neighbor cell
                const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                const unsigned int mapping_index_neigh_cell = 0;
                const dealii::FESystem<dim,dim> &neigh_fe_ref = DGBase<dim,real>::fe_collection[fe_index_neigh_cell];
                const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                // Local rhs contribution from neighbor
                dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                const real penalty1 = deg1sq / vol_div_facearea1;
                const real penalty2 = deg2sq / vol_div_facearea2;
                
                real penalty = 0.5 * ( penalty1 + penalty2 );
                //penalty = 1;//99;

                   // #if 0
                    //for test case
                   // penalty = this->penalty;
                    penalty = DGBase<dim,real>::penalty;
                   // #endif
                    fe_values_collection_volume_neigh.reinit (neighbor_cell, quad_index, mapping_index, fe_index_neigh_cell);
                    const dealii::FEValues<dim,dim> &fe_values_volume_neigh = fe_values_collection_volume_neigh.get_present_fe_values();
                if ( compute_dRdW ) {
                    assemble_face_term_implicit (
                            fe_values_face_int, fe_values_face_ext,
                            fe_values_volume, fe_values_volume_neigh,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                } else {
                    assemble_face_term_explicit (
                            fe_values_face_int, fe_values_face_ext,
                            fe_values_volume, fe_values_volume_neigh,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                }

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    DGBase<dim,real>::right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                }
            } else {
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces
            }

        } // end of face loop

        for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
            DGBase<dim,real>::right_hand_side(current_dofs_indices[i]) += current_cell_rhs(i);
        }

    } // end of cell loop
    DGBase<dim,real>::right_hand_side.compress(dealii::VectorOperation::add);
    if ( compute_dRdW ) DGBase<dim,real>::system_matrix.compress(dealii::VectorOperation::add);

} // end of assemble_system_explicit ()


template class DGStrong <PHILIP_DIM, 1, double>;
template class DGStrong <PHILIP_DIM, 2, double>;
template class DGStrong <PHILIP_DIM, 3, double>;
template class DGStrong <PHILIP_DIM, 4, double>;
template class DGStrong <PHILIP_DIM, 5, double>;

} // PHiLiP namespace

