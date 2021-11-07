#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Models Base Class
//================================================================
template <int dim, int nstate, typename real>
ModelBase<dim, nstate, real>::ModelBase()
{
    // Nothing to do here so far
}
//----------------------------------------------------------------

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, double >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class ModelBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace