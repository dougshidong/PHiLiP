#include "ADTypes.hpp"

#include "viscous_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

// Protyping low level functions
template<int nstate, typename real>
std::array<real, nstate> array_average(
    const std::array<real, nstate> &array1,
    const std::array<real, nstate> &array2)
{
    std::array<real,nstate> array_average;
    for (int s=0; s<nstate; s++) {
        array_average[s] = 0.5*(array1[s] + array2[s]);
    }
    return array_average;
}

template<int nstate, int dim, typename real>
std::array<dealii::Tensor<1,dim,real>, nstate> array_average(
    const std::array<dealii::Tensor<1,dim,real>,nstate> &array1,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &array2)
{
    std::array<dealii::Tensor<1,dim,real>,nstate> array_average;
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<dim; d++) {
            array_average[s][d] = 0.5*(array1[s][d] + array2[s][d]);
        }
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
        for (int d=0; d<dim; d++) {
         array_jump[s][d] = (array1[s] - array2[s])*normal1[d];
        }
    }
    return array_jump;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> CentralViscousNumericalFlux<dim,nstate,real>
::evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    std::array<real,nstate> soln_avg = array_average<nstate,real>(soln_int, soln_ext);

    return soln_avg;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> CentralViscousNumericalFlux<dim,nstate,real>
::evaluate_auxiliary_flux (
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index_,
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext_,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext_,
    const std::array<real, nstate> &filtered_soln_int,
    const std::array<real, nstate> &filtered_soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_ext_,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary,
    const int boundary_type) const
{
    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;

    real artificial_diss_coeff_ext;
    dealii::types::global_dof_index neighbor_cell_index;
    std::array<dealii::Tensor<1,dim,real>, nstate> soln_grad_ext;
    std::array<dealii::Tensor<1,dim,real>, nstate> filtered_soln_grad_ext;
    if (on_boundary) {
        // Following the the boundary treatment given by 
        // Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods, Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        // Details given on page 93
        artificial_diss_coeff_ext = artificial_diss_coeff_int;
        neighbor_cell_index = current_cell_index;
        soln_grad_ext = soln_grad_int;
        //soln_grad_ext = soln_grad_ext_;
        filtered_soln_grad_ext = filtered_soln_grad_int;
    } else {
        artificial_diss_coeff_ext = artificial_diss_coeff_ext_;
        neighbor_cell_index = neighbor_cell_index_;
        soln_grad_ext = soln_grad_ext_;
        filtered_soln_grad_ext = filtered_soln_grad_ext_;
    }

    // {{A*grad_u}}
    std::array<real,nstate> phys_flux_int_dot_n, phys_flux_ext_dot_n;
    phys_flux_int_dot_n = pde_physics->dissipative_flux_dot_normal (soln_int, soln_grad_int, filtered_soln_int, filtered_soln_grad_int, on_boundary, current_cell_index, normal_int, boundary_type);
    phys_flux_ext_dot_n = pde_physics->dissipative_flux_dot_normal (soln_ext, soln_grad_ext, filtered_soln_ext, filtered_soln_grad_ext, on_boundary, neighbor_cell_index, normal_int, boundary_type);
    const std::array<real,nstate> phys_flux_avg_dot_n = array_average<nstate,real>(phys_flux_int_dot_n,phys_flux_ext_dot_n);

    std::array<real,nstate> auxiliary_flux_dot_n = phys_flux_avg_dot_n;

    if (artificial_diss_coeff_int > 1e-13 || artificial_diss_coeff_ext > 1e-13) {
        ArrayTensor1 artificial_phys_flux_int, artificial_phys_flux_ext;

        // {{A*grad_u}}
        artificial_phys_flux_int = artificial_dissip->calc_artificial_dissipation_flux (soln_int, soln_grad_int, artificial_diss_coeff_int);
        artificial_phys_flux_ext = artificial_dissip->calc_artificial_dissipation_flux (soln_ext, soln_grad_ext, artificial_diss_coeff_ext);
        ArrayTensor1 artificial_phys_flux_avg = array_average<nstate,dim,real>(artificial_phys_flux_int, artificial_phys_flux_ext);

        // {{A}}*[[u]]
        ArrayTensor1 soln_jump     = array_jump<dim,nstate,real>(soln_int, soln_ext, normal_int);
        ArrayTensor1 artificial_A_jumpu_int, artificial_A_jumpu_ext;
        artificial_A_jumpu_int = artificial_dissip->calc_artificial_dissipation_flux (soln_int, soln_jump, artificial_diss_coeff_int);
        artificial_A_jumpu_ext = artificial_dissip->calc_artificial_dissipation_flux (soln_ext, soln_jump, artificial_diss_coeff_ext);
        const ArrayTensor1 artificial_A_jumpu_avg = array_average<nstate,dim,real>(artificial_A_jumpu_int, artificial_A_jumpu_ext);

        for (int s=0; s<nstate; s++) {
            //auxiliary_flux_dot_n[s] += (artificial_phys_flux_avg[s] - penalty * artificial_A_jumpu_avg[s]) * normal_int;
            real arti = 0.0;
            for (int d=0; d<dim; ++d) {
                arti += (artificial_phys_flux_avg[s][d] - penalty * artificial_A_jumpu_avg[s][d]) * normal_int[d];
            }
            auxiliary_flux_dot_n[s] += arti;
        }
    }

    return auxiliary_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> SymmetricInternalPenalty<dim,nstate,real>
::evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    std::array<real,nstate> soln_avg = array_average<nstate,real>(soln_int, soln_ext);

    return soln_avg;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> SymmetricInternalPenalty<dim,nstate,real>
::evaluate_auxiliary_flux (
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index_,
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext_,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext_,
    const std::array<real, nstate> &filtered_soln_int,
    const std::array<real, nstate> &filtered_soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_ext_,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary,
    const int boundary_type) const
{
    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;

    real artificial_diss_coeff_ext;
    dealii::types::global_dof_index neighbor_cell_index;
    std::array<dealii::Tensor<1,dim,real>, nstate> soln_grad_ext;
    std::array<dealii::Tensor<1,dim,real>, nstate> filtered_soln_grad_ext;
    if (on_boundary) {
        // Following the the boundary treatment given by 
        // Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods, Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        // Details given on page 93
        artificial_diss_coeff_ext = artificial_diss_coeff_int;
        neighbor_cell_index = current_cell_index;
        // soln_grad_ext = soln_grad_int;
        soln_grad_ext = soln_grad_ext_;
        // filtered_soln_grad_ext = filtered_soln_grad_int;
    } else {
        artificial_diss_coeff_ext = artificial_diss_coeff_ext_;
        neighbor_cell_index = neighbor_cell_index_;
        soln_grad_ext = soln_grad_ext_;
        filtered_soln_grad_ext = filtered_soln_grad_ext_;
    }

    // {{A*grad_u}}
    std::array<real,nstate> phys_flux_int_dot_n, phys_flux_ext_dot_n;
    phys_flux_int_dot_n = pde_physics->dissipative_flux_dot_normal (soln_int, soln_grad_int, filtered_soln_int, filtered_soln_grad_int, on_boundary, current_cell_index, normal_int, boundary_type);
    phys_flux_ext_dot_n = pde_physics->dissipative_flux_dot_normal (soln_ext, soln_grad_ext, filtered_soln_ext, filtered_soln_grad_ext, on_boundary, neighbor_cell_index, normal_int, boundary_type);
    const std::array<real,nstate> phys_flux_avg_dot_n = array_average<nstate,real>(phys_flux_int_dot_n,phys_flux_ext_dot_n);

    // {{A}}*[[u]]
    ArrayTensor1 soln_jump     = array_jump<dim,nstate,real>(soln_int, soln_ext, normal_int);
    ArrayTensor1 filtered_soln_jump = array_jump<dim,nstate,real>(filtered_soln_int, filtered_soln_ext, normal_int);
    std::array<real,nstate> A_jumpu_int_dot_n, A_jumpu_ext_dot_n;
    A_jumpu_int_dot_n = pde_physics->dissipative_flux_dot_normal (soln_int, soln_jump, filtered_soln_int, filtered_soln_jump, on_boundary, current_cell_index, normal_int, boundary_type);
    A_jumpu_ext_dot_n = pde_physics->dissipative_flux_dot_normal (soln_ext, soln_jump, filtered_soln_ext, filtered_soln_jump, on_boundary, neighbor_cell_index, normal_int, boundary_type);
    const std::array<real,nstate> A_jumpu_avg_dot_n = array_average<nstate,real>(A_jumpu_int_dot_n, A_jumpu_ext_dot_n);

    std::array<real,nstate> auxiliary_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        auxiliary_flux_dot_n[s] = phys_flux_avg_dot_n[s] - penalty * A_jumpu_avg_dot_n[s];
    }

    if (artificial_diss_coeff_int > 1e-13 || artificial_diss_coeff_ext > 1e-13) {
        ArrayTensor1 artificial_phys_flux_int, artificial_phys_flux_ext;

        // {{A*grad_u}}
        artificial_phys_flux_int = artificial_dissip->calc_artificial_dissipation_flux (soln_int, soln_grad_int, artificial_diss_coeff_int);
        artificial_phys_flux_ext = artificial_dissip->calc_artificial_dissipation_flux (soln_ext, soln_grad_ext, artificial_diss_coeff_ext);
        ArrayTensor1 artificial_phys_flux_avg = array_average<nstate,dim,real>(artificial_phys_flux_int, artificial_phys_flux_ext);

        // {{A}}*[[u]]
        ArrayTensor1 artificial_A_jumpu_int, artificial_A_jumpu_ext;
        artificial_A_jumpu_int = artificial_dissip->calc_artificial_dissipation_flux (soln_int, soln_jump, artificial_diss_coeff_int);
        artificial_A_jumpu_ext = artificial_dissip->calc_artificial_dissipation_flux (soln_ext, soln_jump, artificial_diss_coeff_ext);
        const ArrayTensor1 artificial_A_jumpu_avg = array_average<nstate,dim,real>(artificial_A_jumpu_int, artificial_A_jumpu_ext);

        for (int s=0; s<nstate; s++) {
            //auxiliary_flux_dot_n[s] += (artificial_phys_flux_avg[s] - penalty * artificial_A_jumpu_avg[s]) * normal_int;
            real arti = 0.0;
            for (int d=0; d<dim; ++d) {
                arti += (artificial_phys_flux_avg[s][d] - penalty * artificial_A_jumpu_avg[s][d]) * normal_int[d];
            }
            auxiliary_flux_dot_n[s] += arti;
        }
    }

    return auxiliary_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> BassiRebay2<dim,nstate,real>
::evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    std::array<real,nstate> soln_avg = array_average<nstate,real>(soln_int, soln_ext);

    return soln_avg;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> BassiRebay2<dim,nstate,real>
::evaluate_auxiliary_flux (
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index_,
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext_,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext_,
    const std::array<real, nstate> &filtered_soln_int,
    const std::array<real, nstate> &filtered_soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &filtered_soln_grad_ext_,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary,
    const int boundary_type) const
{
    using ArrayTensor1 = std::array<dealii::Tensor<1,dim,real>, nstate>;

    (void) on_boundary;
    (void) penalty;

    real artificial_diss_coeff_ext;
    dealii::types::global_dof_index neighbor_cell_index;
    std::array<dealii::Tensor<1,dim,real>, nstate> soln_grad_ext;
    std::array<dealii::Tensor<1,dim,real>, nstate> filtered_soln_grad_ext;
    //if (on_boundary) {
    //    // Following the the boundary treatment given by 
    //    // Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods, Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
    //    // Details given on page 93
    //    artificial_diss_coeff_ext = artificial_diss_coeff_int;
    //    neighbor_cell_index = current_cell_index;
    //    soln_grad_ext = soln_grad_int;
    //    //soln_grad_ext = soln_grad_ext_;
    //    filtered_soln_grad_ext = filtered_soln_grad_int;
    //} else {
        artificial_diss_coeff_ext = artificial_diss_coeff_ext_;
        neighbor_cell_index = neighbor_cell_index_;
        soln_grad_ext = soln_grad_ext_;
        filtered_soln_grad_ext = filtered_soln_grad_ext_;
    //}

    // {{A*grad_u}}
    std::array<real,nstate> phys_flux_int_dot_n, phys_flux_ext_dot_n;
    phys_flux_int_dot_n = pde_physics->dissipative_flux_dot_normal (soln_int, soln_grad_int, filtered_soln_int, filtered_soln_grad_int, on_boundary, current_cell_index, normal_int, boundary_type);
    phys_flux_ext_dot_n = pde_physics->dissipative_flux_dot_normal (soln_ext, soln_grad_ext, filtered_soln_ext, filtered_soln_grad_ext, on_boundary, neighbor_cell_index, normal_int, boundary_type);
    const std::array<real,nstate> phys_flux_avg_dot_n = array_average<nstate,real>(phys_flux_int_dot_n,phys_flux_ext_dot_n);

    std::array<real,nstate> auxiliary_flux_dot_n = phys_flux_avg_dot_n;

    if (artificial_diss_coeff_int > 1e-13 || artificial_diss_coeff_ext > 1e-13) {
        ArrayTensor1 artificial_phys_flux_int, artificial_phys_flux_ext;

        // {{A*grad_u}}
        artificial_phys_flux_int = artificial_dissip->calc_artificial_dissipation_flux (soln_int, soln_grad_int, artificial_diss_coeff_int);
        artificial_phys_flux_ext = artificial_dissip->calc_artificial_dissipation_flux (soln_ext, soln_grad_ext, artificial_diss_coeff_ext);
        ArrayTensor1 artificial_phys_flux_avg = array_average<nstate,dim,real>(artificial_phys_flux_int, artificial_phys_flux_ext);

        for (int s=0; s<nstate; s++) {
            real arti = 0.0;
            for (int d=0; d<dim; ++d) {
                arti += artificial_phys_flux_avg[s][d] * normal_int[d];
            }
            auxiliary_flux_dot_n[s] += arti;
        }
    }

    return auxiliary_flux_dot_n;
}

// Instantiation
template class NumericalFluxDissipative<PHILIP_DIM, 1, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 2, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 3, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 4, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 5, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 6, double>;
template class NumericalFluxDissipative<PHILIP_DIM, 1, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 2, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 3, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 4, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 5, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 6, FadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 1, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 2, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 3, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 4, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 5, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 6, RadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 1, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 2, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 3, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 4, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 5, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 6, FadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 1, RadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 2, RadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 3, RadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 4, RadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 5, RadFadType >;
template class NumericalFluxDissipative<PHILIP_DIM, 6, RadFadType >;


template class SymmetricInternalPenalty<PHILIP_DIM, 1, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 6, double>;
template class SymmetricInternalPenalty<PHILIP_DIM, 1, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 6, FadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 1, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 6, RadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 1, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 6, FadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 1, RadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 2, RadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 3, RadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 4, RadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 5, RadFadType >;
template class SymmetricInternalPenalty<PHILIP_DIM, 6, RadFadType >;

template class BassiRebay2<PHILIP_DIM, 1, double>;
template class BassiRebay2<PHILIP_DIM, 2, double>;
template class BassiRebay2<PHILIP_DIM, 3, double>;
template class BassiRebay2<PHILIP_DIM, 4, double>;
template class BassiRebay2<PHILIP_DIM, 5, double>;
template class BassiRebay2<PHILIP_DIM, 6, double>;
template class BassiRebay2<PHILIP_DIM, 1, FadType >;
template class BassiRebay2<PHILIP_DIM, 2, FadType >;
template class BassiRebay2<PHILIP_DIM, 3, FadType >;
template class BassiRebay2<PHILIP_DIM, 4, FadType >;
template class BassiRebay2<PHILIP_DIM, 5, FadType >;
template class BassiRebay2<PHILIP_DIM, 6, FadType >;
template class BassiRebay2<PHILIP_DIM, 1, RadType >;
template class BassiRebay2<PHILIP_DIM, 2, RadType >;
template class BassiRebay2<PHILIP_DIM, 3, RadType >;
template class BassiRebay2<PHILIP_DIM, 4, RadType >;
template class BassiRebay2<PHILIP_DIM, 5, RadType >;
template class BassiRebay2<PHILIP_DIM, 6, RadType >;
template class BassiRebay2<PHILIP_DIM, 1, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 2, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 3, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 4, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 5, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 6, FadFadType >;
template class BassiRebay2<PHILIP_DIM, 1, RadFadType >;
template class BassiRebay2<PHILIP_DIM, 2, RadFadType >;
template class BassiRebay2<PHILIP_DIM, 3, RadFadType >;
template class BassiRebay2<PHILIP_DIM, 4, RadFadType >;
template class BassiRebay2<PHILIP_DIM, 5, RadFadType >;
template class BassiRebay2<PHILIP_DIM, 6, RadFadType >;

template class CentralViscousNumericalFlux<PHILIP_DIM, 1, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 2, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 3, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 4, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 5, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 6, double>;
template class CentralViscousNumericalFlux<PHILIP_DIM, 1, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 2, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 3, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 4, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 5, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 6, FadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 1, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 2, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 3, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 4, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 5, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 6, RadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 1, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 2, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 3, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 4, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 5, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 6, FadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 1, RadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 2, RadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 3, RadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 4, RadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 5, RadFadType >;
template class CentralViscousNumericalFlux<PHILIP_DIM, 6, RadFadType >;

} // NumericalFlux namespace
} // PHiLiP namespace
