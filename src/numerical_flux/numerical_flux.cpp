#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include "numerical_flux.h"
#include "viscous_numerical_flux.h"
#include "split_form_numerical_flux.h"

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


template <int dim, int nstate, typename real>
NumericalFluxConvective<dim,nstate,real>*
NumericalFluxFactory<dim, nstate, real>
::create_convective_numerical_flux(
    AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    if(conv_num_flux_type == AllParam::lax_friedrichs) {
        return new LaxFriedrichs<dim, nstate, real>(physics_input);
    }
    else if (conv_num_flux_type == AllParam::split_form) {
    	return new SplitFormNumFlux<dim, nstate, real>(physics_input);
    }

    return nullptr;
}
template <int dim, int nstate, typename real>
NumericalFluxDissipative<dim,nstate,real>*
NumericalFluxFactory<dim, nstate, real>
::create_dissipative_numerical_flux(
    AllParam::DissipativeNumericalFlux diss_num_flux_type,
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    if(diss_num_flux_type == AllParam::symm_internal_penalty) {
        return new SymmetricInternalPenalty<dim, nstate, real>(physics_input);
    }

    return nullptr;
}

template <int dim, int nstate, typename real>
NumericalFluxConvective<dim,nstate,real>::~NumericalFluxConvective() {}

template<int dim, int nstate, typename real>
std::array<real, nstate> LaxFriedrichs<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = pde_physics->convective_flux (soln_int);
    conv_phys_flux_ext = pde_physics->convective_flux (soln_ext);
    
    RealArrayVector flux_avg = array_average<nstate, dealii::Tensor<1,dim,real>> (conv_phys_flux_int, conv_phys_flux_ext);

    const real conv_max_eig_int = pde_physics->max_convective_eigenvalue(soln_int);
    const real conv_max_eig_ext = pde_physics->max_convective_eigenvalue(soln_ext);
    const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = flux_avg[s]*normal_int - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
    }

    return numerical_flux_dot_n;
}

// Instantiation
template class NumericalFluxConvective<PHILIP_DIM, 1, double>;
template class NumericalFluxConvective<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 2, double>;
template class NumericalFluxConvective<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 3, double>;
template class NumericalFluxConvective<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 4, double>;
template class NumericalFluxConvective<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class NumericalFluxConvective<PHILIP_DIM, 5, double>;
template class NumericalFluxConvective<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

template class LaxFriedrichs<PHILIP_DIM, 1, double>;
template class LaxFriedrichs<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 2, double>;
template class LaxFriedrichs<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 3, double>;
template class LaxFriedrichs<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 4, double>;
template class LaxFriedrichs<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class LaxFriedrichs<PHILIP_DIM, 5, double>;
template class LaxFriedrichs<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;


template class NumericalFluxFactory<PHILIP_DIM, 1, double>;
template class NumericalFluxFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 2, double>;
template class NumericalFluxFactory<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 3, double>;
template class NumericalFluxFactory<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 4, double>;
template class NumericalFluxFactory<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class NumericalFluxFactory<PHILIP_DIM, 5, double>;
template class NumericalFluxFactory<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;


} // NumericalFlux namespace
} // PHiLiP namespace
