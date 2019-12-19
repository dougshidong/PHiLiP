#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include "viscous_numerical_flux.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

// Protyping low level functions
template<int nstate, typename real_tensor>
std::array<real_tensor, nstate> array_average(
    const std::array<real_tensor, nstate> &array1,
    const std::array<real_tensor, nstate> &array2)
{
    std::array<real_tensor,nstate> array_average;
    for (int s=0; s<nstate; s++) {
        array_average[s] = 0.5*(array1[s] + array2[s]);
    }
    return array_average;
}

template<int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> array_jump(
    const std::array<real, nstate> &array1,
    const std::array<real, nstate> &array2,
    const dealii::Tensor<1,dim,real> &normal1)
{
    std::array<dealii::Tensor<1,dim,real>,nstate> array_jump;
    for (int s=0; s<nstate; s++) {
        array_jump[s] = (array1[s] - array2[s])*normal1;
    }
    return array_jump;
}

template <int dim, int nstate, typename real>
NumericalFluxDissipative<dim,nstate,real>::~NumericalFluxDissipative() {}


//template<int dim, int nstate, typename real>
//void NumericalFluxDissipative<dim,nstate,real>
//::evaluate_auxiliary (
//    const std::vector<std::array<real, nstate>> &soln_nodal_face_int,
//    const std::vector<std::array<real, nstate>> &soln_nodal_face_ext,
//    const std::vector<std::array<real, nstate>> &soln_nodal_face_flux,
//    const std::vector<std::array<real, nstate>> &soln_grad_nodal_volume_int,
//    const std::vector<std::array<real, nstate>> &soln_grad_nodal_volume_ext,
//    const std::vector<dealii::Tensor<1,dim,real>> &normal_int,
//    std::vector<std::array<real, nstate>> &auxiliary_modal_int,
//    std::vector<std::array<real, nstate>> &auxiliary_modal_ext) const
//{
//    // Need some kind of assert to confirm that A and A_transpose correspond
//    // to the current interior and exterior cell

//    // Interpolate solution to the face quadrature points
//    std::vector< ADArray > soln_int(n_face_quad_pts);
//    std::vector< ADArray > soln_ext(n_face_quad_pts);

//    std::vector< ADArrayVector > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
//    std::vector< ADArrayVector > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros
//    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

//        const dealii::Tensor<1,dim,ADtype> normal_int = normals_int[iquad];

//        for (int istate=0; istate<nstate; ++istate) {
//            soln_int[iquad][istate]      = 0;
//            soln_ext[iquad][istate]      = 0;
//            soln_grad_int[iquad][istate] = 0;
//            soln_grad_ext[iquad][istate] = 0;

//            // Interpolate solution to face
//            for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
//                soln_int[iquad][istate]      += current_solution_ad[itrial][istate] * fe_values_int->shape_value(itrial, iquad);
//                soln_grad_int[iquad][istate] += current_solution_ad[itrial][istate] * fe_values_int->shape_grad(itrial, iquad);
//            }
//            for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
//                soln_ext[iquad][istate]      += neighbor_solution_ad[itrial][istate] * fe_values_ext->shape_value(itrial, iquad);
//                soln_grad_ext[iquad][istate] += neighbor_solution_ad[itrial][istate] * fe_values_ext->shape_grad(itrial, iquad);
//            }
//        }
//        normal_int_numerical_flux[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

//    }
//}


//template<int dim, int nstate, typename real>
//void NumericalFluxDissipative<dim,nstate,real>
//::assemble_auxiliary_cell (
//    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int) const
//{
//}

template<int dim, int nstate, typename real>
std::array<real, nstate> SymmetricInternalPenalty<dim,nstate,real>
::evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    std::array<real,nstate> soln_avg = array_average<nstate,real>(soln_int, soln_ext);

    //std::array<dealii::Tensor<2,dim,real>,nstate> diffusion_matrix_int = 
    //    pde_physics->diffusion_matrix(soln_int);

    //std::array<dealii::Tensor<2,dim,real>,nstate> diffusion_matrix_ext = 
    //    pde_physics->diffusion_matrix(soln_ext);

    return soln_avg;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> SymmetricInternalPenalty<dim,nstate,real>
::evaluate_auxiliary_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary) const
{
    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;

    if (on_boundary) {
        // Following the the boundary treatment given by 
        // Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods, Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        // Details given on page 93
        const std::array<real, nstate> soln_bc = soln_ext;
        //const std::array<dealii::Tensor<1,dim,real>, nstate> soln_grad_bc = soln_grad_ext;
        const ArrayTensor1 phys_flux_bc = pde_physics->dissipative_flux (soln_bc, soln_grad_int);

        const ArrayTensor1 soln_jump    = array_jump<dim,nstate,real>(soln_int, soln_bc, normal_int);
        const ArrayTensor1 Abc_jumpu    = pde_physics->dissipative_flux (soln_bc, soln_jump);
        std::array<real,nstate> auxiliary_flux_dot_n;
        for (int s=0; s<nstate; s++) {
            auxiliary_flux_dot_n[s] = (phys_flux_bc[s] - penalty * Abc_jumpu[s]) * normal_int;
        }
        return auxiliary_flux_dot_n;
    } 

    ArrayTensor1 phys_flux_int, phys_flux_ext;

    // {{A*grad_u}}
    phys_flux_int = pde_physics->dissipative_flux (soln_int, soln_grad_int);
    phys_flux_ext = pde_physics->dissipative_flux (soln_ext, soln_grad_ext);
    ArrayTensor1 phys_flux_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(phys_flux_int, phys_flux_ext);

    // {{A}}*[[u]]
    ArrayTensor1 soln_jump     = array_jump<dim,nstate,real>(soln_int, soln_ext, normal_int);
    ArrayTensor1 A_jumpu_int, A_jumpu_ext;
    A_jumpu_int = pde_physics->dissipative_flux (soln_int, soln_jump);
    A_jumpu_ext = pde_physics->dissipative_flux (soln_ext, soln_jump);
    const ArrayTensor1 A_jumpu_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(A_jumpu_int, A_jumpu_ext);


    std::array<real,nstate> auxiliary_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * A_jumpu_avg[s]) * normal_int;
        //if (on_boundary) auxiliary_flux_dot_n[s] = (phys_flux_ext[s] - penalty * A_jumpu_int[s]) * normal_int;
        //auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * soln_jump[s]) * normal_int;
    }
    return auxiliary_flux_dot_n;

}

//Bassi Rebay 2 numerical flux
template<int dim, int nstate, typename real>
std::array<real, nstate> BassiRebay2<dim,nstate,real>
::evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    std::array<real,nstate> soln_avg = array_average<nstate,real>(soln_int, soln_ext);

    //std::array<dealii::Tensor<2,dim,real>,nstate> diffusion_matrix_int = 
    //    pde_physics->diffusion_matrix(soln_int);

    //std::array<dealii::Tensor<2,dim,real>,nstate> diffusion_matrix_ext = 
    //    pde_physics->diffusion_matrix(soln_ext);

    return soln_avg;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> BassiRebay2<dim,nstate,real>
::evaluate_auxiliary_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary) const
{
    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;

    if (on_boundary) {
        // Following the the boundary treatment given by 
        // Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods, Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        // Details given on page 93
        const std::array<real, nstate> soln_bc = soln_ext;
        //const std::array<dealii::Tensor<1,dim,real>, nstate> soln_grad_bc = soln_grad_ext;
        const ArrayTensor1 phys_flux_bc = pde_physics->dissipative_flux (soln_bc, soln_grad_int);

        const ArrayTensor1 soln_jump    = array_jump<dim,nstate,real>(soln_int, soln_bc, normal_int);
        const ArrayTensor1 Abc_jumpu    = pde_physics->dissipative_flux (soln_bc, soln_jump);
        std::array<real,nstate> auxiliary_flux_dot_n;
        for (int s=0; s<nstate; s++) {
            auxiliary_flux_dot_n[s] = (phys_flux_bc[s] - penalty * Abc_jumpu[s]) * normal_int;
        }
        return auxiliary_flux_dot_n;
    } 

    ArrayTensor1 phys_flux_int, phys_flux_ext;

    // {{A*grad_u}}
    phys_flux_int = pde_physics->dissipative_flux (soln_int, soln_grad_int);
    phys_flux_ext = pde_physics->dissipative_flux (soln_ext, soln_grad_ext);
    ArrayTensor1 phys_flux_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(phys_flux_int, phys_flux_ext);

    // {{A}}*[[u]]
    ArrayTensor1 soln_jump     = array_jump<dim,nstate,real>(soln_int, soln_ext, normal_int);
    ArrayTensor1 A_jumpu_int, A_jumpu_ext;
    A_jumpu_int = pde_physics->dissipative_flux (soln_int, soln_jump);
    A_jumpu_ext = pde_physics->dissipative_flux (soln_ext, soln_jump);
    const ArrayTensor1 A_jumpu_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(A_jumpu_int, A_jumpu_ext);


    std::array<real,nstate> auxiliary_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * A_jumpu_avg[s]) * normal_int;
        //if (on_boundary) auxiliary_flux_dot_n[s] = (phys_flux_ext[s] - penalty * A_jumpu_int[s]) * normal_int;
        //auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * soln_jump[s]) * normal_int;
    }
    return auxiliary_flux_dot_n;

}
//template<int dim, int nstate, typename real>
//std::array<real, nstate> BassiRebay2<dim,nstate,real>
//::evaluate_auxiliary_flux (
//    const std::array<real, nstate> &soln_int,
//    const std::array<real, nstate> &soln_ext,
//    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
//    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
//    const dealii::Tensor<1,dim,real> &normal_int,
//    const real &penalty) const
//{
//    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;
//    ArrayTensor1 phys_flux_int, phys_flux_ext;

//    // {{A*grad_u}}
//    pde_physics->dissipative_flux (soln_int, soln_grad_int, phys_flux_int);
//    pde_physics->dissipative_flux (soln_ext, soln_grad_ext, phys_flux_ext);
//    ArrayTensor1 phys_flux_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(phys_flux_int, phys_flux_ext);

//    // {{A}}*[[u]]
//    ArrayTensor1 soln_jump     = array_jump<dim,nstate,real>(soln_int, soln_ext, normal_int);
//    ArrayTensor1 A_jumpu_int, A_jumpu_ext;
//    pde_physics->dissipative_flux (soln_int, soln_jump, A_jumpu_int);
//    pde_physics->dissipative_flux (soln_ext, soln_jump, A_jumpu_ext);
//    const ArrayTensor1 A_jumpu_avg = array_average<nstate,dealii::Tensor<1,dim,real>>(A_jumpu_int, A_jumpu_ext);


//    std::array<real,nstate> auxiliary_flux_dot_n;
//    for (int s=0; s<nstate; s++) {
//        auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * A_jumpu_avg[s]) * normal_int;
//        //auxiliary_flux_dot_n[s] = (phys_flux_avg[s] - penalty * soln_jump[s]) * normal_int;
//    }
//    return auxiliary_flux_dot_n;

//}




// Instantiation
template class NumericalFluxDissipative<PHILIP_DIM, 1, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class NumericalFluxDissipative<PHILIP_DIM, 2, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class NumericalFluxDissipative<PHILIP_DIM, 3, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class NumericalFluxDissipative<PHILIP_DIM, 4, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class NumericalFluxDissipative<PHILIP_DIM, 5, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

template class SymmetricInternalPenalty<PHILIP_DIM, 1, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

template class BassiRebay2<PHILIP_DIM, 1, double>;
template class BassiRebay2<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class BassiRebay2<PHILIP_DIM, 2, double>;
template class BassiRebay2<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class BassiRebay2<PHILIP_DIM, 3, double>;
template class BassiRebay2<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class BassiRebay2<PHILIP_DIM, 4, double>;
template class BassiRebay2<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class BassiRebay2<PHILIP_DIM, 5, double>;
template class BassiRebay2<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;
} // NumericalFlux namespace
} // PHiLiP namespace
