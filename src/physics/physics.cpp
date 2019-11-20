#include <assert.h>
#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::PhysicsBase() 
    : manufactured_solution_function(nstate)
{
    const double pi = atan(1)*4.0;
    const double ee = exp(1);

    // Some constants used to define manufactured solution
    velo_x = 1.1; velo_y = -pi/ee; velo_z = ee/pi;
    //velo_x = 2.0; velo_y = 2.0; velo_z = ee/pi;
    diff_coeff = 0.1*pi/ee;
    diff_coeff = 0.1;
    diff_coeff = 0.5;

    // Anisotropic diffusion matrix
    //A11 =   9; A12 =  -2; A13 =  -6;
    //A21 =   3; A22 =  20; A23 =   4;
    //A31 =  -2; A32 = 0.5; A33 =   8;

    diffusion_tensor[0][0] = 12;
    if (dim>=2) {
        diffusion_tensor[0][1] = -2;
        diffusion_tensor[1][0] = 3;
        diffusion_tensor[1][1] = 20;
    }
    if (dim>=3) {
        diffusion_tensor[0][2] = -6;
        diffusion_tensor[1][2] = -4;
        diffusion_tensor[2][0] = -2;
        diffusion_tensor[2][1] = 0.5;
        diffusion_tensor[2][2] = 8;
    }

    diffusion_tensor[0][0] = 12;
    if (dim>=2) {
        diffusion_tensor[0][1] = 3;
        diffusion_tensor[1][0] = 3;
        diffusion_tensor[1][1] = 20;
    }
    if (dim>=3) {
        diffusion_tensor[0][2] = -2;
        diffusion_tensor[2][0] = 2;
        diffusion_tensor[2][1] = 5;
        diffusion_tensor[1][2] = -5;
        diffusion_tensor[2][2] = 18;
    }
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            diffusion_tensor[i][j] = (i==j) ? 1.0 : 0.0;
}

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::~PhysicsBase() {}

template <int dim, int nstate, typename real>
void PhysicsBase<dim,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, double> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    std::array<real,nstate> boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
    for (int s=0; s<nstate; s++) {
        boundary_values[s] = this->manufactured_solution_function.value (pos, s);
        boundary_gradients[s] = this->manufactured_solution_function.gradient (pos, s);
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
template class PhysicsBase < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 2, double >;
template class PhysicsBase < PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 3, double >;
template class PhysicsBase < PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 4, double >;
template class PhysicsBase < PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 5, double >;
template class PhysicsBase < PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 8, double >;
template class PhysicsBase < PHILIP_DIM, 8, Sacado::Fad::DFad<double> >;

} // Physics namespace
} // PHiLiP namespace

