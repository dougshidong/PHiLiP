#include "ADTypes.hpp"

#include "advection_spacetime.h"

namespace PHiLiP {
namespace Physics {

template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
template class ConvectionDiffusion < PHILIP_DIM, 1, FadType>;
template class ConvectionDiffusion < PHILIP_DIM, 1, RadType>;
template class ConvectionDiffusion < PHILIP_DIM, 1, FadFadType>;
template class ConvectionDiffusion < PHILIP_DIM, 1, RadFadType>;
} // Physics namespace
} // PHiLiP namespace
