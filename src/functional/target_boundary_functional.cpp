#include "target_boundary_functional.h"

namespace PHiLiP {
#if PHILIP_SPECIES==1
    // Define a sequence of indices representing the range [1, 5]
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)

    // Define a macro to instantiate Target Boundary Functional for a specific index
    #define INSTANTIATE_FUNCTIONAL(r, data, index) \
        template class TargetBoundaryFunctional <PHILIP_DIM, PHILIP_SPECIES, index, double>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNCTIONAL, _, POSSIBLE_NSTATE)
#endif
} // PHiLiP namespace
