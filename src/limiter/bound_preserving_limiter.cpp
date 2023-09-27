#include "bound_preserving_limiter.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
// Constructor
template <int dim, typename real>
BoundPreservingLimiter<dim, real>::BoundPreservingLimiter(
    const int nstate_input,
    const Parameters::AllParameters* const parameters_input)
    : nstate(nstate_input)
    , all_parameters(parameters_input) {}


template <int dim, int nstate, typename real>
BoundPreservingLimiterState<dim, nstate, real>::BoundPreservingLimiterState(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiter<dim, real>::BoundPreservingLimiter(nstate, parameters_input)
{}

template <int dim, int nstate, typename real>
std::array<real, nstate> BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg(
    std::array<std::vector<real>, nstate> soln_at_q,
    const unsigned int n_quad_pts,
    const std::vector<real>& quad_weights)
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

template class BoundPreservingLimiter <PHILIP_DIM, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 1, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 2, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 3, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 4, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 5, double>;
template class BoundPreservingLimiterState <PHILIP_DIM, 6, double>;
} // PHiLiP namespace