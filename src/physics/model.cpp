#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Models Base Class
//================================================================
template <int dim, int nstate, typename real>
ModelBase<dim, nstate, real>::ModelBase(
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input):
        manufactured_solution_function(manufactured_solution_function_input)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
ModelBase<dim,nstate,real>::~ModelBase() {}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim, nstate, real>
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
    for (int s=0; s<nstate; s++) {
        boundary_values[s] = this->manufactured_solution_function->value (pos, s);
        boundary_gradients[s] = this->manufactured_solution_function->gradient (pos, s);
    }

    for (int istate=0; istate<dim+2; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
    }
    for (int istate=dim+2; istate<nstate; ++istate) {

        std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow) { // Dirichlet boundary condition
            // soln_bc[istate] = boundary_values[istate];
            // soln_grad_bc[istate] = soln_grad_int[istate];

            soln_bc[istate] = boundary_values[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];

        } else { // Neumann boundary condition
            // //soln_bc[istate] = soln_int[istate];
            // //soln_bc[istate] = boundary_values[istate];
            // soln_bc[istate] = -soln_int[istate]+2*boundary_values[istate];

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


//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelBase<PHILIP_DIM, 1, double>;
template class ModelBase<PHILIP_DIM, 2, double>;
template class ModelBase<PHILIP_DIM, 3, double>;
template class ModelBase<PHILIP_DIM, 4, double>;
template class ModelBase<PHILIP_DIM, 5, double>;
template class ModelBase<PHILIP_DIM, 6, double>;
template class ModelBase<PHILIP_DIM, 8, double>;

template class ModelBase<PHILIP_DIM, 1, FadType>;
template class ModelBase<PHILIP_DIM, 2, FadType>;
template class ModelBase<PHILIP_DIM, 3, FadType>;
template class ModelBase<PHILIP_DIM, 4, FadType>;
template class ModelBase<PHILIP_DIM, 5, FadType>;
template class ModelBase<PHILIP_DIM, 6, FadType>;
template class ModelBase<PHILIP_DIM, 8, FadType>;

template class ModelBase<PHILIP_DIM, 1, RadType>;
template class ModelBase<PHILIP_DIM, 2, RadType>;
template class ModelBase<PHILIP_DIM, 3, RadType>;
template class ModelBase<PHILIP_DIM, 4, RadType>;
template class ModelBase<PHILIP_DIM, 5, RadType>;
template class ModelBase<PHILIP_DIM, 6, RadType>;
template class ModelBase<PHILIP_DIM, 8, RadType>;

template class ModelBase<PHILIP_DIM, 1, FadFadType>;
template class ModelBase<PHILIP_DIM, 2, FadFadType>;
template class ModelBase<PHILIP_DIM, 3, FadFadType>;
template class ModelBase<PHILIP_DIM, 4, FadFadType>;
template class ModelBase<PHILIP_DIM, 5, FadFadType>;
template class ModelBase<PHILIP_DIM, 6, FadFadType>;
template class ModelBase<PHILIP_DIM, 8, FadFadType>;

template class ModelBase<PHILIP_DIM, 1, RadFadType>;
template class ModelBase<PHILIP_DIM, 2, RadFadType>;
template class ModelBase<PHILIP_DIM, 3, RadFadType>;
template class ModelBase<PHILIP_DIM, 4, RadFadType>;
template class ModelBase<PHILIP_DIM, 5, RadFadType>;
template class ModelBase<PHILIP_DIM, 6, RadFadType>;
template class ModelBase<PHILIP_DIM, 8, RadFadType>;

} // Physics namespace
} // PHiLiP namespace