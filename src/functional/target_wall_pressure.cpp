#include "target_wall_pressure.hpp"

namespace PHiLiP {

#if PHILIP_SPECIES==1
template class TargetWallPressure <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2, double>;
#endif
} // PHiLiP namespace
