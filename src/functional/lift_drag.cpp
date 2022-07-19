#include "lift_drag.hpp"

namespace PHiLiP {

#if PHILIP_DIM != 1
template class LiftDragFunctional <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace

