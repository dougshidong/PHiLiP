#include "bound_preserving_limiter.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
// Constructor
template <int dim, int nspecies, typename real>
BoundPreservingLimiter<dim, nspecies, real>::BoundPreservingLimiter(
    const int nstate_input,
    const Parameters::AllParameters* const parameters_input)
    : nstate(nstate_input)
    , all_parameters(parameters_input) {}


template <int dim, int nspecies, int nstate, typename real>
BoundPreservingLimiterState<dim, nspecies, nstate, real>::BoundPreservingLimiterState(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiter<dim, nspecies, real>::BoundPreservingLimiter(nstate, parameters_input)
{}

template <int dim, int nspecies, int nstate, typename real>
std::array<real, nstate> BoundPreservingLimiterState<dim, nspecies, nstate, real>::get_soln_cell_avg(
    const std::array<std::vector<real>, nstate>&        soln_at_q,
    const unsigned int                                  n_quad_pts,
    const std::vector<real>&                            quad_weights)
{
    std::array<real, nstate> soln_cell_avg;
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        soln_cell_avg[istate] = 0;
    }

    // Apply integral for solution cell average (dealii quadrature operates from [0,1])
    for (unsigned int istate = 0; istate < nstate; ++istate) {
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            soln_cell_avg[istate] += quad_weights[iquad]
                * soln_at_q[istate][iquad];
        }
    }

    return soln_cell_avg;
}

#if PHILIP_SPECIES==1
    template class BoundPreservingLimiter <PHILIP_DIM, PHILIP_SPECIES, double>;
    // Define a sequence of nstate in the range [1, 6]
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)

    // Define a macro to instantiate Limiter Function for a specific nstate
    #define INSTANTIATE_LIMITER(r, data, nstate) \
        template class BoundPreservingLimiterState <PHILIP_DIM, PHILIP_SPECIES, nstate, double>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_LIMITER, _, POSSIBLE_NSTATE)
#endif
} // PHiLiP namespace