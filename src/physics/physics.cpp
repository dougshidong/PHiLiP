#include <assert.h>
#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::PhysicsBase(
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input):
        manufactured_solution_function(manufactured_solution_function_input)
{
    // if provided with a null ptr, give it the default manufactured solution
    // currently only necessary for the unit test
    if(!manufactured_solution_function)
        manufactured_solution_function = std::make_shared<ManufacturedSolutionSine<dim,real>>(nstate);

    // anisotropic diffusion matrix
    diffusion_tensor[0][0] = input_diffusion_tensor[0][0];
    if constexpr(dim >= 2) {
        diffusion_tensor[0][1] = input_diffusion_tensor[0][1];
        diffusion_tensor[1][0] = input_diffusion_tensor[1][0];
        diffusion_tensor[1][1] = input_diffusion_tensor[1][1];
    }
    if constexpr(dim >= 3) {
        diffusion_tensor[0][2] = input_diffusion_tensor[0][2];
        diffusion_tensor[2][0] = input_diffusion_tensor[2][0];
        diffusion_tensor[1][2] = input_diffusion_tensor[1][2];
        diffusion_tensor[2][1] = input_diffusion_tensor[2][1];
        diffusion_tensor[2][2] = input_diffusion_tensor[2][2];
    }
}

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::~PhysicsBase() {}
/*
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsBase<dim,nstate,real>
::artificial_dissipative_flux (
    const real viscosity_coefficient,
    const std::array<real,nstate> &,//solution,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    for (int i=0; i<nstate; i++) {
        for (int d=0; d<dim; d++) {
            diss_flux[i][d] = -viscosity_coefficient*(solution_gradient[i][d]);
        }
    }
    return diss_flux;
}
*/
template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsBase<dim,nstate,real>
::artificial_source_term (
    const real viscosity_coefficient,
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    
    dealii::Tensor<2,dim,double> artificial_diffusion_tensor;
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            artificial_diffusion_tensor[i][j] = (i==j) ? 1.0 : 0.0;

    for (int istate=0; istate<nstate; istate++) {
        dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function->hessian (pos, istate);
        //source[istate] = -viscosity_coefficient*scalar_product(artificial_diffusion_tensor,manufactured_hessian);
        source[istate] = 0.0;
        for (int dr=0; dr<dim; ++dr) {
            for (int dc=0; dc<dim; ++dc) {
                source[istate] += artificial_diffusion_tensor[dr][dc] * manufactured_hessian[dr][dc];
            }
        }
        source[istate] *= -viscosity_coefficient;
    }
    return source;
}

template <int dim, int nstate, typename real>
void PhysicsBase<dim,nstate,real>
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

    for (int istate=0; istate<nstate; ++istate) {

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

template <int dim, int nstate, typename real>
dealii::Vector<double> PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
    const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
    const dealii::Tensor<1,dim>                  &/*normals*/,
    const dealii::Point<dim>                  &/*evaluation_points*/) const
{
    dealii::Vector<double> computed_quantities(nstate);
    for (unsigned int s=0; s<nstate; ++s) {
        computed_quantities(s) = uh(s);
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real>
dealii::Vector<double> PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_scalar (
    const double              &uh,
    const dealii::Tensor<1,dim> &/*duh*/,
    const dealii::Tensor<2,dim> &/*dduh*/,
    const dealii::Tensor<1,dim> &/*normals*/,
    const dealii::Point<dim>    &/*evaluation_points*/) const
{
    assert(nstate == 1);
    dealii::Vector<double> computed_quantities(nstate);
    for (unsigned int s=0; s<nstate; ++s) {
        computed_quantities(s) = uh;
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<std::string> PhysicsBase<dim,nstate,real> ::post_get_names () const
{
    std::vector<std::string> names;
    for (unsigned int s=0; s<nstate; ++s) {
        std::string varname = "state" + dealii::Utilities::int_to_string(s,1);
        names.push_back(varname);
    }
    return names;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> PhysicsBase<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    for (unsigned int s=0; s<nstate; ++s) {
        interpretation.push_back (DCI::component_is_scalar);
    }
    return interpretation;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags PhysicsBase<dim,nstate,real>
::post_get_needed_update_flags () const
{
    return dealii::update_values;
}

template class PhysicsBase < PHILIP_DIM, 1, double >;
template class PhysicsBase < PHILIP_DIM, 2, double >;
template class PhysicsBase < PHILIP_DIM, 3, double >;
template class PhysicsBase < PHILIP_DIM, 4, double >;
template class PhysicsBase < PHILIP_DIM, 5, double >;
template class PhysicsBase < PHILIP_DIM, 8, double >;

template class PhysicsBase < PHILIP_DIM, 1, FadType >;
template class PhysicsBase < PHILIP_DIM, 2, FadType >;
template class PhysicsBase < PHILIP_DIM, 3, FadType >;
template class PhysicsBase < PHILIP_DIM, 4, FadType >;
template class PhysicsBase < PHILIP_DIM, 5, FadType >;
template class PhysicsBase < PHILIP_DIM, 8, FadType >;

template class PhysicsBase < PHILIP_DIM, 1, RadType >;
template class PhysicsBase < PHILIP_DIM, 2, RadType >;
template class PhysicsBase < PHILIP_DIM, 3, RadType >;
template class PhysicsBase < PHILIP_DIM, 4, RadType >;
template class PhysicsBase < PHILIP_DIM, 5, RadType >;
template class PhysicsBase < PHILIP_DIM, 8, RadType >;

template class PhysicsBase < PHILIP_DIM, 1, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 2, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 3, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 4, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 5, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 8, FadFadType >;

template class PhysicsBase < PHILIP_DIM, 1, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 2, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 3, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 4, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 5, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 8, RadFadType >;

} // Physics namespace
} // PHiLiP namespace
