#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "convection_diffusion.h"

namespace PHiLiP {
namespace Physics {

template <int nstate, typename real>
std::array<real,nstate> stdvector_to_stdarray(const std::vector<real> vector)
{
    std::array<real,nstate> array;
    for (int i=0; i<nstate; i++) { array[i] = vector[i]; }
    return array;
}

template <int dim, int nstate, typename real>
void ConvectionDiffusion<dim,nstate,real>
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

        std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow || hasDiffusion) { // Dirichlet boundary condition
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
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const dealii::Tensor<1,dim,real> velocity_field = advection_speed();
    for (int i=0; i<nstate; ++i) {
        conv_flux[i] = velocity_field * solution[i];
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_numerical_split_flux (
    const std::array<real,nstate> &soln1,
    const std::array<real,nstate> &soln2) const
{
    std::array<real,nstate> arr_avg;
    for (int i = 0 ; i < nstate; ++i) {
        arr_avg[i] = (soln1[i] + soln2[i])/2.;
    }
    return convective_flux(arr_avg);
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ConvectionDiffusion<dim,nstate,real>
::advection_speed () const
{
    dealii::Tensor<1,dim,real> advection_speed;
    if (hasConvection) {
        if(dim >= 1) advection_speed[0] = linear_advection_velocity[0];
        if(dim >= 2) advection_speed[1] = linear_advection_velocity[1];
        if(dim >= 3) advection_speed[2] = linear_advection_velocity[2];
    } else {
        const real zero = 0.0;
        if(dim >= 1) advection_speed[0] = zero;
        if(dim >= 2) advection_speed[1] = zero;
        if(dim >= 3) advection_speed[2] = zero;
    }
    return advection_speed;
}
template <int dim, int nstate, typename real>
real ConvectionDiffusion<dim,nstate,real>
::diffusion_coefficient () const
{
    if(hasDiffusion) return diffusion_scaling_coeff;
    const real zero = 0.0;
    return zero;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*solution*/,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    const dealii::Tensor<1,dim,real> advection_speed = this->advection_speed();
    for (int i=0; i<nstate; i++) {
        eig[i] = advection_speed*normal;
    }
    return eig;
}

template <int dim, int nstate, typename real>
real ConvectionDiffusion<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*soln*/) const
{
    const dealii::Tensor<1,dim,real> advection_speed = this->advection_speed();
    real max_eig = 0;
    for (int i=0; i<dim; i++) {
        max_eig = std::max(max_eig,std::abs(advection_speed[0]));
    }
    return max_eig;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    const real diff_coeff = diffusion_coefficient();
    for (int i=0; i<nstate; i++) {
  for (int d1=0; d1<dim; d1++) {
   diss_flux[i][d1] = 0.0;
   for (int d2=0; d2<dim; d2++) {
     diss_flux[i][d1] += -diff_coeff*(this->diffusion_tensor[d1][d2]*solution_gradient[i][d2]);
   }
  }
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> ConvectionDiffusion<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    const dealii::Tensor<1,dim,real> velocity_field = this->advection_speed();
    const real diff_coeff = diffusion_coefficient();

    for (int istate=0; istate<nstate; istate++) {
        dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, istate);
            // dealii::Tensor<1,dim,real> manufactured_gradient_fd = this->manufactured_solution_function.gradient_fd (pos, istate);
            // std::cout<<"FD" <<std::endl;
            // std::cout<<manufactured_gradient_fd <<std::endl;
            // std::cout<<"AN" <<std::endl;
            // std::cout<<manufactured_gradient <<std::endl;
            // std::cout<<"DIFF" <<std::endl;
            // std::cout<<manufactured_gradient - manufactured_gradient_fd <<std::endl;
        dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function->hessian (pos, istate);
            // dealii::SymmetricTensor<2,dim,real> manufactured_hessian_fd = this->manufactured_solution_function.hessian_fd (pos, istate);
            // std::cout<<"FD" <<std::endl;
            // std::cout<<manufactured_hessian_fd <<std::endl;
            // std::cout<<"AN" <<std::endl;
            // std::cout<<manufactured_hessian <<std::endl;
            // std::cout<<"DIFF" <<std::endl;
            // std::cout<<manufactured_hessian - manufactured_hessian_fd <<std::endl;
        source[istate] = velocity_field*manufactured_gradient;
        source[istate] += -diff_coeff*scalar_product((this->diffusion_tensor),manufactured_hessian);
    }
    return source;
}

template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
template class ConvectionDiffusion < PHILIP_DIM, 2, double >;
template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
template class ConvectionDiffusion < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;
template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<Sacado::Fad::DFad<double>>  >;
template class ConvectionDiffusion < PHILIP_DIM, 2, Sacado::Fad::DFad<Sacado::Fad::DFad<double>>  >;
template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Rad::ADvar<Sacado::Fad::DFad<double>>  >;
template class ConvectionDiffusion < PHILIP_DIM, 2, Sacado::Rad::ADvar<Sacado::Fad::DFad<double>>  >;

} // Physics namespace
} // PHiLiP namespace


