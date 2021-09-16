#include "ADTypes.hpp"

#include "burgers_rewienski.h"

namespace PHiLiP {
    namespace Physics {

        template <int dim, int nstate, typename real>
        void BurgersRewienski<dim,nstate,real>
        ::boundary_face_values (
                const int /*boundary_type*/,
                const dealii::Point<dim, real> &pos,
                const dealii::Tensor<1,dim,real> &normal_int,
                const std::array<real,nstate> &soln_int,
                const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
                std::array<real,nstate> &soln_bc,
                std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
        {
            std::array<real,nstate> boundary_values;
            std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
            for (int i=0; i<nstate; i++) {
                boundary_values[i] = this->manufactured_solution_function->value (pos, i);
                boundary_gradients[i] = this->manufactured_solution_function->gradient (pos, i);
            }

            for (int istate=0; istate<nstate; ++istate) {

                std::array<real,nstate> characteristic_dot_n = this->convective_eigenvalues(boundary_values, normal_int);
                const bool inflow = (characteristic_dot_n[istate] <= 0.);

                if (inflow || this->hasDiffusion) { // Dirichlet boundary condition
                    // soln_bc[istate] = boundary_values[istate];
                    // soln_grad_bc[istate] = soln_grad_int[istate];

                    //soln_bc[istate] = boundary_values[istate];
                    soln_bc[istate] = 2.5;
                    soln_grad_bc[istate] = soln_grad_int[istate];

                } else { // Neumann boundary condition
                    // //soln_bc[istate] = soln_int[istate];
                    // //soln_bc[istate] = boundary_values[istate];
                    //soln_bc[istate] = -soln_int[istate]+2*boundary_values[istate];
                    soln_bc[istate] = soln_int[istate];

                    // **************************************************************************************************************
                    // Note I don't know how to properly impose the soln_grad_bc to obtain an adjoint consistent scheme
                    // Currently, Neumann boundary conditions are only imposed for the linear advection
                    // Therefore, soln_grad_bc does not affect the solution
                    // **************************************************************************************************************
                    soln_grad_bc[istate] = soln_grad_int[istate];
                    //soln_grad_bc[istate] = boundary_gradients[istate];
                    //soln_grad_bc[istate] = -soln_grad_int[istate]+2*boundary_gradients[istate];
                }
            }
        }

        template <int dim, int nstate, typename real>
        std::array<real,nstate> BurgersRewienski<dim,nstate,real>
        ::source_term (
                const dealii::Point<dim,real> &pos,
                const std::array<real,nstate> &/*solution*/) const
        {
            std::array<real,nstate> source;
            const real diff_coeff = this->diffusion_coefficient();

            for (int istate=0; istate<nstate; istate++) {
                source[istate] = 0.0;
                dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, istate);
                dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function->hessian (pos, istate);
                for (int d=0;d<dim;++d) {
                    real manufactured_solution = this->manufactured_solution_function->value (pos, d);
                    source[istate] += 0.5*manufactured_solution*manufactured_gradient[d];
                }
                //source[istate] += -diff_coeff*scalar_product((this->diffusion_tensor),manufactured_hessian);
                real hess = 0.0;
                for (int dr=0; dr<dim; ++dr) {
                    for (int dc=0; dc<dim; ++dc) {
                        hess += (this->diffusion_tensor)[dr][dc] * manufactured_hessian[dr][dc];
                    }
                }
                source[istate] += -diff_coeff*hess;
            }
            for (int istate=0; istate<nstate; istate++) {
                real manufactured_solution = this->manufactured_solution_function->value (pos, istate);
                real divergence = 0.0;
                for (int d=0;d<dim;++d) {
                    dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, d);
                    divergence += manufactured_gradient[d];
                }
                source[istate] += 0.5*manufactured_solution*divergence;
            }

            return source;
        }

        template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, double >;
        template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, FadType  >;
        template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, RadType  >;
        template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, FadFadType >;
        template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, RadFadType >;

    } // Physics namespace
} // PHiLiP namespace



