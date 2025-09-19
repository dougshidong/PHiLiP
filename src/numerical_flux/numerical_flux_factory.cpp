#include <boost/preprocessor/seq/for_each.hpp>
#include "numerical_flux_factory.hpp"

#include "convective_numerical_flux.hpp"
#include "ADTypes.hpp"
#include "physics/physics_model.h"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

template <int dim, int nspecies, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nspecies, nstate, real>
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
    // checks if weak dg is being run with two point flux                          
    const bool is_two_point_conv = ((conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux) ||
                                    (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_lax_friedrichs_dissipation) || 
                                    (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_roe_dissipation) || 
                                    (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_l2roe_dissipation));
    if(is_two_point_conv && physics_input->all_parameters->use_split_form == false ) {
        std::cout << "two point flux and not using split form are not compatible, please use another Convective Numerical Flux" << std::endl;
        std::abort();
    }

    if (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::central_flux) {
        if constexpr (nstate<=5) {
            return std::make_unique< Central<dim, nstate, real> > (physics_input);
        }
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
        if constexpr (nstate<=5) {
            return std::make_unique< EntropyConserving<dim, nstate, real> > (physics_input);
        }
    } 
    else if (conv_num_flux_type == AllParam::ConvectiveNumericalFlux::two_point_flux_with_lax_friedrichs_dissipation) {
        if constexpr (nstate<=5) {
            return std::make_unique< EntropyConservingWithLaxFriedrichsDissipation<dim, nstate, real> > (physics_input);
        }
    } 
    else {
        (void) pde_type;
        (void) model_type;
    }

    std::cout << "Invalid convective numerical flux and/or invalid added Riemann solver dissipation type." << std::endl;
    return nullptr;
}

template <int dim, int nspecies, int nstate, typename real>
std::unique_ptr< NumericalFluxConvective<dim,nstate,real> >
NumericalFluxFactory<dim, nspecies, nstate, real>
::create_euler_based_convective_numerical_flux(
    const AllParam::ConvectiveNumericalFlux conv_num_flux_type,
    const AllParam::PartialDifferentialEquation pde_type,
    const AllParam::ModelType model_type,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
{
    using PDE_enum   = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum = Parameters::AllParameters::ModelType;
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> euler_based_physics_to_be_passed = physics_input;

    if(pde_type!=PDE_enum::euler && 
        pde_type!=PDE_enum::navier_stokes && 
        !(pde_type==PDE_enum::physics_model && model_type==Model_enum::large_eddy_simulation)) 
    {
        std::cout << "Invalid convective numerical flux for pde_type. Aborting..." << std::endl;
        std::abort();
    }

#if PHILIP_DIM==3
    if((pde_type==PDE_enum::physics_model && 
        model_type==Model_enum::large_eddy_simulation)) 
    {
        if constexpr (dim+2==nstate) {
            std::shared_ptr<Physics::PhysicsModel<dim,nspecies,dim+2,real,dim+2>> physics_model = std::dynamic_pointer_cast<Physics::PhysicsModel<dim,nspecies,dim+2,real,dim+2>>(physics_input);
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

template <int dim, int nspecies, int nstate, typename real>
std::unique_ptr< NumericalFluxDissipative<dim,nstate,real> >
NumericalFluxFactory<dim, nspecies, nstate, real>
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

// Define a sequence of indices representing the range [1, 8]
#define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)(7)(8)

// Define a macro to instantiate MyTemplate for a specific index
#define INSTANTIATE_DOUBLE(r, data, index) \
    template class NumericalFluxFactory <PHILIP_DIM, PHILIP_SPECIES, index, double>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_DOUBLE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_FADTYPE(r, data, index) \
    template class NumericalFluxFactory <PHILIP_DIM, PHILIP_SPECIES, index, FadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_RADTYPE(r, data, index) \
    template class NumericalFluxFactory <PHILIP_DIM, PHILIP_SPECIES, index, RadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_RADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_FADFADTYPE(r, data, index) \
    template class NumericalFluxFactory <PHILIP_DIM, PHILIP_SPECIES, index, FadFadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FADFADTYPE, _, POSSIBLE_NSTATE)

#define INSTANTIATE_RADFADTYPE(r, data, index) \
    template class NumericalFluxFactory <PHILIP_DIM, PHILIP_SPECIES, index, RadFadType>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_RADFADTYPE, _, POSSIBLE_NSTATE)
} // NumericalFlux namespace
} // PHiLiP namespace
