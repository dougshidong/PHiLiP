#include "ADTypes.hpp"

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
        conv_flux[i] = 0.0;
        for (int d=0; d<dim; ++d) {
//std::cout<<" vel "<<velocity_field[d]<<" dir "<<d<<std::endl;
            conv_flux[i][d] += velocity_field[d] * solution[i];
        }
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
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_surface_numerical_split_flux (
                const std::array< dealii::Tensor<1,dim,real>, nstate > &/*surface_flux*/,
                const std::array< dealii::Tensor<1,dim,real>, nstate > &flux_interp_to_surface) const
{
    return flux_interp_to_surface;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ConvectionDiffusion<dim,nstate,real>
::advection_speed () const
{
    dealii::Tensor<1,dim,real> advection_speed;
    if (hasConvection) {
        if(dim >= 1) advection_speed[0] = linear_advection_velocity[0];
       // if(dim >= 1) advection_speed[0] = 1.0;
        if(dim >= 2) advection_speed[1] = linear_advection_velocity[1];
       // if(dim >= 2) advection_speed[1] = 1.0;
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
    real eig_value = 0.0;
    for (int d=0; d<dim; ++d) {
        eig_value += advection_speed[d]*normal[d];
    }
    for (int i=0; i<nstate; i++) {
        eig[i] = eig_value;
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
        real abs_adv = abs(advection_speed[i]);
        max_eig = std::max(max_eig,abs_adv);
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
    const std::array<real,nstate> &/*solution*/,
    const real current_time) const
{
    std::array<real,nstate> source;
    const dealii::Tensor<1,dim,real> velocity_field = this->advection_speed();
    const real diff_coeff = diffusion_coefficient();


    using TestType = Parameters::AllParameters::TestType;

    if(this->test_type == TestType::convection_diffusion_periodicity){
        for(int istate =0; istate<nstate; istate++){
            source[istate] = 0.0;
            const double pi = atan(1)*4.0;
            real sine_term = 1.0;
            for(int idim=0; idim<dim; idim++){
                sine_term *= sin(pi * pos[idim]);
            }
            source[istate] += (- diff_coeff) * exp(-diff_coeff * current_time) * sine_term;//the unsteady term
            for(int idim=0; idim<dim; idim++){//laplacian term
                source[istate] += diff_coeff * pow(pi,2) * exp(-diff_coeff * current_time)
                                * this->diffusion_tensor[idim][idim] * sine_term;
            }
            for(int idim=0; idim<dim; idim++){//cross terms
                for(int jdim=0; jdim<dim; jdim++){
                    if(idim != jdim){
                        real cross_term = cos(pi*pos[idim]) * cos(pi*pos[jdim]);
                        if(dim == 3){
                            int kdim = 3 - idim - jdim;  
                            cross_term *= sin(pi*pos[kdim]);
                        }
                        source[istate] += - diff_coeff * pow(pi,2) * exp(-diff_coeff * current_time)
                                        * this->diffusion_tensor[idim][jdim] * cross_term;
                    }
                }
            } 
        }
    }
    else{
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
         
            //source[istate] = velocity_field*manufactured_gradient;
            real grad = 0.0;
            for (int d=0; d<dim; ++d) {
                grad += velocity_field[d] * manufactured_gradient[d];
            }
            source[istate] = grad;
         
            real hess = 0.0;
            for (int dr=0; dr<dim; ++dr) {
                for (int dc=0; dc<dim; ++dc) {
                    hess += (this->diffusion_tensor)[dr][dc] * manufactured_hessian[dr][dc];
                }
            }
            source[istate] += -diff_coeff*hess;
        }
    }
    return source;
}

template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
template class ConvectionDiffusion < PHILIP_DIM, 2, double >;
template class ConvectionDiffusion < PHILIP_DIM, 3, double >;
template class ConvectionDiffusion < PHILIP_DIM, 4, double >;
template class ConvectionDiffusion < PHILIP_DIM, 5, double >;

template class ConvectionDiffusion < PHILIP_DIM, 1, FadType>;
template class ConvectionDiffusion < PHILIP_DIM, 2, FadType>;
template class ConvectionDiffusion < PHILIP_DIM, 3, FadType>;
template class ConvectionDiffusion < PHILIP_DIM, 4, FadType>;
template class ConvectionDiffusion < PHILIP_DIM, 5, FadType>;

template class ConvectionDiffusion < PHILIP_DIM, 1, RadType>;
template class ConvectionDiffusion < PHILIP_DIM, 2, RadType>;
template class ConvectionDiffusion < PHILIP_DIM, 3, RadType>;
template class ConvectionDiffusion < PHILIP_DIM, 4, RadType>;
template class ConvectionDiffusion < PHILIP_DIM, 5, RadType>;

template class ConvectionDiffusion < PHILIP_DIM, 1, FadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 2, FadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 3, FadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 4, FadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 5, FadFadType>;

template class ConvectionDiffusion < PHILIP_DIM, 1, RadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 2, RadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 3, RadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 4, RadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 5, RadFadType>;

} // Physics namespace
} // PHiLiP namespace


