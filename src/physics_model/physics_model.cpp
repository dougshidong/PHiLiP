#include <cmath>
#include <vector>

#include "physics/physics.h"
#include "physics/euler.h"
#include "physics/navier_stokes.h"

#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace PhysicsModel {

//================================================================
// Physics Models Base Class
//================================================================
template <int dim, int nstate, typename real>
PhysicsModelBase<dim, nstate, real>::PhysicsModelBase()
{
    // Nothing to do here so far
}
//----------------------------------------------------------------

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, double >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class PhysicsModelBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // PhysicsModel namespace
} // PHiLiP namespace