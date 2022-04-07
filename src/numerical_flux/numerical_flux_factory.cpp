#include "numerical_flux_factory.hpp"

#include "ADTypes.hpp"
#include "split_form_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_convective_numerical_flux(
    AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    // checks if conv_numerical_flux_type exists only for Euler equations
    const bool is_euler_flux = ((conv_num_flux_type == AllParam::roe) ||
                                (conv_num_flux_type == AllParam::l2roe));

    if(conv_num_flux_type == AllParam::lax_friedrichs) {
        return std::make_unique< LaxFriedrichs<dim, nstate, real> > (physics_input);
    } else if(is_euler_flux) {
        return create_euler_convective_numerical_flux(conv_num_flux_type, physics_input);
    } else if (conv_num_flux_type == AllParam::split_form) {
        return std::make_unique< SplitFormNumFlux<dim, nstate, real> > (physics_input);
    }

    std::cout << "Invalid convective numerical flux" << std::endl;
    return nullptr;
}

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_euler_convective_numerical_flux(
    AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
#if PHILIP_DIM==3
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_to_be_passed = physics_input;
    try { 
        // check if physics_input is a PhysicsModel
        const auto &physics_model = dynamic_cast<const Physics::PhysicsModel<dim,dim+2,real,dim+2>&>(*physics_input);
        // check if the physics_baseline in the PhysicsModel object is Euler or derived from Euler (e.g. NavierStokes)
        const auto &physics_baseline = dynamic_cast<const Physics::Euler<dim,dim+2,real>&>(*physics_model->physics_baseline);
        
        if(physics_model->n_model_equations > 0) {
            std::cout << "Error: Cannot create_euler_convective_numerical_flux() for nstate_baseline_physics != nstate." << std::endl;
            std::abort();
        } else {
            // pass the baseline physics object of type Euler
            physics_to_be_passed = physics_model->physics_baseline;
        }
    } catch (std::bad_cast&) {
        std::cout << "Invalid convective numerical flux for physics_model and/or corresponding baseline_physics_type" << std::endl;
        std::abort();
    }
    if(conv_num_flux_type == AllParam::roe) {
        if constexpr (dim+2==nstate) return std::make_unique< RoePike<dim, nstate, real> > (physics_to_be_passed);
    } else if(conv_num_flux_type == AllParam::l2roe) {
        if constexpr (dim+2==nstate) return std::make_unique< L2Roe<dim, nstate, real> > (physics_to_be_passed);
    }
#else
    if(conv_num_flux_type == AllParam::roe) {
        if constexpr (dim+2==nstate) return std::make_unique< RoePike<dim, nstate, real> > (physics_input);
    } else if(conv_num_flux_type == AllParam::l2roe) {
        if constexpr (dim+2==nstate) return std::make_unique< L2Roe<dim, nstate, real> > (physics_input);
    }
#endif
}

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxDissipative<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_dissipative_numerical_flux(
    AllParam::DissipativeNumericalFlux diss_num_flux_type,
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input, std::shared_ptr<ArtificialDissipationBase<dim, nstate>>  artificial_dissipation_input)
{
    if(diss_num_flux_type == AllParam::symm_internal_penalty) {
        return std::make_unique < SymmetricInternalPenalty<dim, nstate, real> > (physics_input,artificial_dissipation_input);
    } else if(diss_num_flux_type == AllParam::bassi_rebay_2) {
        return std::make_unique < BassiRebay2<dim, nstate, real> > (physics_input,artificial_dissipation_input);
    }

    std::cout << "Invalid dissipative flux" << std::endl;
    return nullptr;
}

template class NumericalFluxFactory<PHILIP_DIM, 1, double>;
template class NumericalFluxFactory<PHILIP_DIM, 2, double>;
template class NumericalFluxFactory<PHILIP_DIM, 3, double>;
template class NumericalFluxFactory<PHILIP_DIM, 4, double>;
template class NumericalFluxFactory<PHILIP_DIM, 5, double>;
template class NumericalFluxFactory<PHILIP_DIM, 1, FadType >;
template class NumericalFluxFactory<PHILIP_DIM, 2, FadType >;
template class NumericalFluxFactory<PHILIP_DIM, 3, FadType >;
template class NumericalFluxFactory<PHILIP_DIM, 4, FadType >;
template class NumericalFluxFactory<PHILIP_DIM, 5, FadType >;
template class NumericalFluxFactory<PHILIP_DIM, 1, RadType >;
template class NumericalFluxFactory<PHILIP_DIM, 2, RadType >;
template class NumericalFluxFactory<PHILIP_DIM, 3, RadType >;
template class NumericalFluxFactory<PHILIP_DIM, 4, RadType >;
template class NumericalFluxFactory<PHILIP_DIM, 5, RadType >;
template class NumericalFluxFactory<PHILIP_DIM, 1, FadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 2, FadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 3, FadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 4, FadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 5, FadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 1, RadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 2, RadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 3, RadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 4, RadFadType >;
template class NumericalFluxFactory<PHILIP_DIM, 5, RadFadType >;


} // NumericalFlux namespace
} // PHiLiP namespace
