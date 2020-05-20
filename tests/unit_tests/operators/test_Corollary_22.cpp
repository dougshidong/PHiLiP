#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

const double TOLERANCE = 1E-6;
using namespace std;

int main (int /*argc*/, char * /*argv*/[])
{

    using real = double;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;

    unsigned int poly_degree = 2;


        //build the fe_collections Lagrange
            dealii::hp::FECollection<dim> fe_collection;

            dealii::hp::QCollection<dim> volume_quadrature_collection_GL;
            unsigned int over_int_GL = poly_degree + 3;

            for (unsigned int degree=0; degree<=poly_degree; ++degree) {

            const dealii::FE_DGQ<dim> fe_dg(degree);
            const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
            fe_collection.push_back (fe_system);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (over_int_GL);
            volume_quadrature_collection_GL.push_back(vol_quad_Gauss_Legendre);
            }

        //build the fe_collections Legendre
            dealii::hp::FECollection<dim> fe_collection_Legendre;

            dealii::hp::QCollection<dim> volume_quadrature_collection_GLL;
            unsigned int over_int_GLL = poly_degree + 3;

            for (unsigned int degree=0; degree<=poly_degree; ++degree) {

            const dealii::FE_DGQLegendre<dim> fe_dgp(degree);
            const dealii::FESystem<dim,dim> fe_system_Legendre(fe_dgp, nstate);
            fe_collection_Legendre.push_back (fe_system_Legendre);
           // dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (over_int_GLL);
           // volume_quadrature_collection_GLL.push_back(vol_quad_Gauss_Lobatto);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (over_int_GL);
            volume_quadrature_collection_GLL.push_back(vol_quad_Gauss_Legendre);
            }


            const unsigned int n_dofs = pow(poly_degree+1,dim); 
            const unsigned int n_quad_GL = pow(over_int_GL,dim);
            const unsigned int n_quad_GLL = pow(over_int_GLL,dim);
            unsigned int istate = 0;

            //get chi and phi: chi-Lagrange phi-Legendre
            dealii::FullMatrix<real> chi(n_quad_GL, n_dofs);
            std::vector<dealii::FullMatrix<real>> dchi_dxi(dim);
            for(int idim=0; idim<dim; idim++){
                dchi_dxi[idim].reinit(n_quad_GL, n_dofs);
            }
            dealii::FullMatrix<real> phi(n_quad_GLL, n_dofs);
            std::vector<dealii::FullMatrix<real>> dphi_dxi(dim);
            for(int idim=0; idim<dim; idim++){
                dphi_dxi[idim].reinit(n_quad_GLL, n_dofs);
            }
            dealii::FullMatrix<real> W_GL(n_quad_GL);
            dealii::FullMatrix<real> W_GLL(n_quad_GLL);
            const std::vector<real> &quad_weights_GL = volume_quadrature_collection_GL[poly_degree].get_weights ();
            const std::vector<real> &quad_weights_GLL = volume_quadrature_collection_GLL[poly_degree].get_weights ();
            dealii::FullMatrix<real> chi_GLL(n_quad_GL, n_dofs);
            dealii::FullMatrix<real> phi_GL(n_quad_GLL, n_dofs);

            //get basis functions
            for(unsigned int idof=0; idof<n_dofs; idof++){
                //get basis evaluated at GL nodes
                for(unsigned int iquad=0; iquad<n_quad_GL; iquad++){
                    const dealii::Point<dim> qpoint  = volume_quadrature_collection_GL[poly_degree].point(iquad);
                    chi[iquad][idof] = fe_collection[poly_degree].shape_value_component(idof,qpoint,istate);
                    phi_GL[iquad][idof] = fe_collection_Legendre[poly_degree].shape_value_component(idof,qpoint,istate);
                }
                //get basis evaluated at GLL nodes
                for(unsigned int iquad=0; iquad<n_quad_GLL; iquad++){
                    const dealii::Point<dim> qpoint  = volume_quadrature_collection_GLL[poly_degree].point(iquad);
                    phi[iquad][idof] = fe_collection_Legendre[poly_degree].shape_value_component(idof,qpoint,istate);
                    chi_GLL[iquad][idof] = fe_collection[poly_degree].shape_value_component(idof,qpoint,istate);
                }
            }
            for(unsigned int iquad=0; iquad<n_quad_GL; iquad++){
                W_GL[iquad][iquad] = quad_weights_GL[iquad];
            }
            for(unsigned int iquad=0; iquad<n_quad_GLL; iquad++){
                W_GLL[iquad][iquad] = quad_weights_GLL[iquad];
            }

            dealii::FullMatrix<real> chi_trans_W_GL(n_dofs,n_quad_GL);
            chi.Tmmult(chi_trans_W_GL, W_GL);//chi^T*W
            dealii::FullMatrix<real> M_GL(n_dofs);
            chi_trans_W_GL.mmult(M_GL, chi);//chit^T*W*chi
            dealii::FullMatrix<real> M_GL_inv(n_dofs);
            M_GL_inv.invert(M_GL);

            dealii::FullMatrix<real> phi_trans_W_GLL(n_dofs,n_quad_GLL);
            phi.Tmmult(phi_trans_W_GLL, W_GLL);
            dealii::FullMatrix<real> M_GLL(n_dofs);
            phi_trans_W_GLL.mmult(M_GLL, phi);
            dealii::FullMatrix<real> M_GLL_inv(n_dofs);
            M_GLL_inv.invert(M_GLL);

            dealii::FullMatrix<real> projection_GL(n_dofs, n_quad_GL);
            M_GL_inv.mmult(projection_GL, chi_trans_W_GL);//M^{-1}*chit^T*W ie/ projection of chi at GL nodes
            dealii::FullMatrix<real> projection_GLL(n_dofs, n_quad_GLL);
            M_GLL_inv.mmult(projection_GLL, phi_trans_W_GLL);// projection of phi at GLL nodes
            
           //get derivatives of basis at respective nodes 
            for (unsigned int iquad=0; iquad<n_quad_GL; ++iquad) {
                for (unsigned int idof=0; idof<n_dofs; ++idof) {
                    dealii::Tensor<1,dim,real> derivative;
                    const dealii::Point<dim> qpoint  = volume_quadrature_collection_GL[poly_degree].point(iquad);
                    derivative = fe_collection[poly_degree].shape_grad_component(idof, qpoint, istate);
                    for (int idim=0; idim<dim; idim++){
                        dchi_dxi[idim][iquad][idof] = derivative[idim];//store dChi/dXi at GL nodes
                    }
                }
            }
            for (unsigned int iquad=0; iquad<n_quad_GLL; ++iquad) {
                for (unsigned int idof=0; idof<n_dofs; ++idof) {
                    dealii::Tensor<1,dim,real> derivative;
                    const dealii::Point<dim> qpoint  = volume_quadrature_collection_GLL[poly_degree].point(iquad);
                    derivative = fe_collection_Legendre[poly_degree].shape_grad_component(idof, qpoint, istate);
                    for (int idim=0; idim<dim; idim++){
                        dphi_dxi[idim][iquad][idof] = derivative[idim];//store dphi/dXi at GLL nodes
                    }
                }
            }


            dealii::FullMatrix<real> D_Lagrange(n_dofs);
            projection_GL.mmult(D_Lagrange, dchi_dxi[0]);//P*dchi/dxi at GL
            dealii::FullMatrix<real> D_Legendre(n_dofs);
            projection_GLL.mmult(D_Legendre, dphi_dxi[0]);//P*dphi/dxi at GLL



        dealii::FullMatrix<real> I_stof(n_dofs);//interps from soln (GLL nodes) to flux (GL nodes)
        projection_GL.mmult(I_stof, phi_GL);//Interp soln to flux from definition evaluated at GL nodes
        dealii::FullMatrix<real> DI_StoF(n_quad_GL, n_dofs);
        D_Lagrange.mmult(DI_StoF, I_stof);//D at GL nodes* I stof
        dealii::FullMatrix<real> I_StoF_Dhat(n_quad_GLL, n_dofs);
        I_stof.mmult(I_StoF_Dhat, D_Legendre);//I stof * \hat{D} at GLL nodes
        printf(" DIstof \n");
        for (unsigned int iquad=0; iquad<n_quad_GL; ++iquad) {
            for (unsigned int idof=0; idof<n_dofs; ++idof) {
                printf(" %g ",DI_StoF[iquad][idof]);
            }
            printf("\n");
        }

        printf(" Istof D\n");
        for (unsigned int iquad=0; iquad<n_quad_GLL; ++iquad) {
            for (unsigned int idof=0; idof<n_dofs; ++idof) {
                printf(" %g ",I_StoF_Dhat[iquad][idof]);
            }
            printf("\n");
        }
    


    return 0;
}

