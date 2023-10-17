#ifndef __BOUND_PRESERVING_LIMITER_FACTORY_H__
#define __BOUND_PRESERVING_LIMITER_FACTORY_H__

#include "parameters/all_parameters.h"
#include "bound_preserving_limiter.h"

namespace PHiLiP {

/// This class creates a new BoundPreservingLimiter object
template <int dim, int nstate, typename real>
class BoundPreservingLimiterFactory
{
public:
    static std::unique_ptr< BoundPreservingLimiter<dim,real> > create_limiter(
        const Parameters::AllParameters *const parameters_input);

    static std::unique_ptr< BoundPreservingLimiter<dim, real> > select_limiter(
        const Parameters::AllParameters* const parameters_input);

};

} // PHiLiP namespace

#endif
