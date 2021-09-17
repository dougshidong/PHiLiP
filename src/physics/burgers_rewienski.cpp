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
                    soln_bc[istate] = sqrt(5); //for testing
                    soln_grad_bc[istate] = soln_grad_int[istate];

                } else { // Neumann boundary condition
                    soln_bc[istate] = soln_int[istate];
                    soln_grad_bc[istate] = soln_grad_int[istate];
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

            for (int istate=0; istate<nstate; istate++) {
                double b = 0.02; //test b = 0.02
                source[istate] = 0.02*exp(b*pos(0));
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



