#ifndef __BOUND_PRESERVING_LIMITER_FACTORY_H__
#define __BOUND_PRESERVING_LIMITER_FACTORY_H__

#include "parameters/all_parameters.h"
#include "bound_preserving_limiter.h"

namespace PHiLiP {
namespace LIMITER {

/// This class creates a new BoundPreservingLimiter object
template <int dim, typename real>
class BoundPreservingLimiterFactory
{
public:
    /// Creates a derived object Operators, but returns it as OperatorsBase.
    /** That way, the caller is agnostic to the number of state variables,
      * poly degree, dofs, etc.*/
    static std::shared_ptr< BoundPreservingLimiterBase<dim,real> >
    create_operators(
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input);

};

} // LIMITER namespace
} // PHiLiP namespace

#endif
