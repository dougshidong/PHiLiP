#include "parameters/all_parameters.h"
#include "bound_preserving_limiter_factory.hpp"
#include "bound_preserving_limiter.h"

namespace PHiLiP {
template <int dim, typename real>
std::shared_ptr< BoundPreservingLimiter<dim,real> >
BoundPreservingLimiterFactory<dim,real>
::create_limiter(
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input)
{
    using limiter_enum = Parameters::AllParameters::LimiterType;
    limiter_enum limiter_type = parameters_input->use_scaling_limiter_type;


    if (nstate_input == 1) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 1, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 1, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 1, real> >(parameters_input);
                break;
            }
        }
    }
    else if (nstate_input == 2) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 2, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 2, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 2, real> >(parameters_input);
                break;
            }
        }
    }
    else if (nstate_input == 3) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 3, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 3, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 3, real> >(parameters_input);
                break;
            }
        }
    }
    else if (nstate_input == 4) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 4, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 4, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 4, real> >(parameters_input);
                break;
            }
        }
    }
    else if (nstate_input == 5) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 5, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 5, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 5, real> >(parameters_input);
                break;
            }
        }
    }
    else if (nstate_input == 6) {
        switch (limiter_type)
        {
            case limiter_enum::none:
            {
                return nullptr;
                break;
            }
            case limiter_enum::maximum_principle:
            {
                return std::make_shared< MaximumPrincipleLimiter<dim, 6, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2010:
            {
                return std::make_shared< PositivityPreservingLimiter<dim, 6, real> >(parameters_input);
                break;
            }
            case limiter_enum::positivity_preserving2011:
            {
                return std::make_shared< PositivityPreservingLimiterRobust<dim, 6, real> >(parameters_input);
                break;
            }
        }
    }
    else {
        std::cout << "Number of states " << nstate_input << "not supported." << std::endl;
        return nullptr;
    }

    assert(0 == 1 && "Cannot create limiter pointer due to an invalid limiter type specified");
    return nullptr;
}

template class BoundPreservingLimiterFactory <PHILIP_DIM, double>;
} // PHiLiP namespace
