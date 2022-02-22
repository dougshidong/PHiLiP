#ifndef __PHILIP_NUMERICAL_FLUX_FACTORY__
#define __PHILIP_NUMERICAL_FLUX_FACTORY__

#include "physics/physics.h"
#include "dg/artificial_dissipation.h"

#include "convective_numerical_flux.hpp"
#include "viscous_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {
/// Creates a NumericalFluxConvective or NumericalFluxDissipative based on input.
template <int dim, int nstate, typename real>
class NumericalFluxFactory
{
public:
    /// Creates convective numerical flux based on input.
    static std::unique_ptr < NumericalFluxConvective<dim,nstate,real> >
        create_convective_numerical_flux
            (AllParam::ConvectiveNumericalFlux conv_num_flux_type,
            std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input);
    /// Creates dissipative numerical flux based on input.
    static std::unique_ptr < NumericalFluxDissipative<dim,nstate,real> >
        create_dissipative_numerical_flux
            (AllParam::DissipativeNumericalFlux diss_num_flux_type,
            std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input, std::shared_ptr<ArtificialDissipationBase<dim, nstate>>  artificial_dissipation_input);
};


} // NumericalFlux namespace
} // PHiLiP namespace

#endif
