#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsModel<dim, nstate, real>::PhysicsModel( 
    Parameters::AllParameters::PartialDifferentialEquation       baseline_physics_type,
    const int                                                    nstate_baseline_physics,
    std::shared_ptr< PHiLiP::PhysicsModelBase<dim,nstate,real> > physics_model_input,
    const dealii::Tensor<2,3,double>                             input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> >    manufactured_solution_function)
    : PhysicsBase<dim,nstate,real>(input_diffusion_tensor,manufactured_solution_function)
    , nstate_baseline_physics(nstate_baseline_physics)
    , n_model_equations(nstate-nstate_baseline_physics)
    , physics_model(physics_model_input)
{
    // Creates the baseline physics
    physics_baseline = PhysicsFactory<dim,real>::create_Physics(parameters_input, baseline_physics_type);
}

// Instantiate explicitly
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, double >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace