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
PhysicsModelBase<dim, nstate, real>::PhysicsModelBase( 
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : PhysicsBase<dim,nstate,real>(input_diffusion_tensor,manufactured_solution_function)
    , turbulent_prandtl_number(turbulent_prandtl_number)
    , nBaselineEquations(nstate-nModelEquations)
    , nModelEquations(nModelEquations)
{
    // static_assert(nstate==dim+2, "Physics::LargeEddySimulationBase() should be created with nstate=dim+2");
    // Nothing to do here so far
}


// Instantiate explicitly
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, double >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace