#include "parameters/all_parameters.h"
#include "bound_preserving_limiter_factory.hpp"
#include "bound_preserving_limiter.h"

namespace PHiLiP {
template <int dim, typename real>
std::shared_ptr< BoundPreservingLimiterBase<dim,real> >//returns type OperatorsBase
BoundPreservingLimiterFactory<dim,real>
::create_limiter(
        const Parameters::AllParameters *const parameters_input)
{
    using limiter_enum = Parameters::AllParameters::LimiterType;
    limiter_enum limiter_type = parameters_input->use_scaling_limiter;

    switch(limiter_type)
    {
        case limiter_enum::none:
        {
            return nullptr;
            break;
        }
        case limiter_enum::maximum_principle:
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, nstate, real> >(parameters_input);
            break;
        }
        case limiter_enum::positivity_preserving2010:
        {
            return std::make_shared< PositivityPreservingLimiter<dim, nstate, real> >(parameters_input);
            break;
        }
        case limiter_enum::positivity_preserving2011:
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, nstate, real> >(parameters_input);
            break;
        }
    }
    assert(0==1 && "Cannot create limiter pointer due to an invalid limiter type specified"); 
    return nullptr;
}

template class BoundPreservingLimiterFactory <PHILIP_DIM, double>;
} // PHiLiP namespace