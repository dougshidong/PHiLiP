#include "numerical_flux_factory.hpp"

#include "convective_numerical_flux.hpp"
#include "ADTypes.hpp"
#include "physics/physics_model.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_convective_numerical_flux(
    const AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    const AllParam::PartialDifferentialEquation pde_type,
    const AllParam::ModelType model_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    // checks if conv_num_flux_type exists only for Euler equations
    const bool is_euler_based = ((conv_num_flux_type == AllParam::ConvectiveNumericalFlux::roe) ||
                                 (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::l2roe) || 
                                 (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_roe_dissipation) || 
                                 (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_l2roe_dissipation));

    if (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::central_flux) {
        return std::make_unique< Central<dim, nstate, real> > (physics_input);
    }
    else if(conv_num_flux_type == AllParam::ConvectiveNumericalFlux::lax_friedrichs) {
        return std::make_unique< LaxFriedrichs<dim, nstate, real> > (physics_input);
    } 
    else if(is_euler_based) {
        if constexpr (dim+2==nstate) {
            return create_euler_based_convective_numerical_flux(conv_num_flux_type, pde_type, model_type, physics_input);
        }
    }
    else if (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux) {
        return std::make_unique< EntropyConserving<dim, nstate, real> > (physics_input);
    } 
    else if (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_lax_friedrichs_dissipation) {
        return std::make_unique< EntropyConservingWithLaxFriedrichsDissipation<dim, nstate, real> > (physics_input);
    } 
    else {
        (void) pde_type;
        (void) model_type;
    }

    std::cout << "Invalid convective numerical flux and/or invalid added Riemann solver dissipation type." << std::endl;
    return nullptr;
}

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_euler_based_convective_numerical_flux(
    const AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    const AllParam::PartialDifferentialEquation pde_type,
    const AllParam::ModelType model_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    using PDE_enum   = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum = Parameters::AllParameters::ModelType;
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> euler_based_physics_to_be_passed = physics_input;

#if PHILIP_DIM==3
    if((pde_type==PDE_enum::physics_model && 
        model_type==Model_enum::large_eddy_simulation)) 
    {
        if constexpr (dim+2==nstate) {
            std::shared_ptr<Physics::PhysicsModel<dim,dim+2,real,dim+2>> physics_model = std::dynamic_pointer_cast<Physics::PhysicsModel<dim,dim+2,real,dim+2>>(physics_input);
            std::shared_ptr<Physics::Euler<dim,dim+2,real>> physics_baseline = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,real>>(physics_model->physics_baseline);
            euler_based_physics_to_be_passed = physics_baseline;
        }
    }
    else if((pde_type==PDE_enum::physics_model && 
             model_type!=Model_enum::large_eddy_simulation)) 
    {
        std::cout << "Invalid convective numerical flux for physics_model and/or corresponding baseline_physics_type" << std::endl;
        if(nstate!=(dim+2)) std::cout << "Error: Cannot create_euler_based_convective_numerical_flux() for nstate_baseline_physics != nstate." << std::endl;
        std::abort();
    }
#endif
    if(conv_num_flux_type == AllParam::ConvectiveNumericalFlux::roe) {
        if constexpr (dim+2==nstate) return std::make_unique< RoePike<dim, nstate, real> > (euler_based_physics_to_be_passed);
    } 
    else if(conv_num_flux_type == AllParam::ConvectiveNumericalFlux::l2roe) {
        if constexpr (dim+2==nstate) return std::make_unique< L2Roe<dim, nstate, real> > (euler_based_physics_to_be_passed);
    } 
    else if(conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_roe_dissipation) {
        if constexpr (dim+2==nstate) return std::make_unique< EntropyConservingWithRoeDissipation<dim, nstate, real> > (euler_based_physics_to_be_passed);
    }
    else if(conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_l2roe_dissipation) {
        if constexpr (dim+2==nstate) return std::make_unique< EntropyConservingWithL2RoeDissipation<dim, nstate, real> > (euler_based_physics_to_be_passed);
    }

    (void) pde_type;
    (void) model_type;

    std::cout << "Invalid Euler based convective numerical flux" << std::endl;
    return nullptr;
}

template <int dim, int nstate, typename real>
std::unique_ptr< NumericalFluxDissipative<dim,nstate,real> >
NumericalFluxFactory<dim, nstate, real>
::create_dissipative_numerical_flux(
    const AllParam::DissipativeNumericalFlux diss_num_flux_type,
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, real>> physics_input,
    std::shared_ptr<ArtificialDissipationBase<dim, nstate>>  artificial_dissipation_input)
{
    if(diss_num_flux_type == AllParam::symm_internal_penalty) {
        return std::make_unique < SymmetricInternalPenalty<dim, nstate, real> > (physics_input,artificial_dissipation_input);
    } else if(diss_num_flux_type == AllParam::bassi_rebay_2) {
        return std::make_unique < BassiRebay2<dim, nstate, real> > (physics_input,artificial_dissipation_input);
    } else if(diss_num_flux_type == AllParam::central_visc_flux) {
        return std::make_unique < CentralViscousNumericalFlux<dim, nstate, real> > (physics_input,artificial_dissipation_input);
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
