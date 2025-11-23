#include "target_boundary_functional.h"

namespace PHiLiP {
#if PHILIP_SPECIES==1
    // Define a sequence of nstate in the range [1, 5]
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)

    // Define a macro to instantiate Target Boundary Functional for a specific nstate
    #define INSTANTIATE_FUNCTIONAL(r, data, nstate) \
        template class TargetBoundaryFunctional <PHILIP_DIM, PHILIP_SPECIES, nstate, double>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNCTIONAL, _, POSSIBLE_NSTATE)
#endif
} // PHiLiP namespace
