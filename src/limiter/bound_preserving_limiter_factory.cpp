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
    bool apply_tvb = parameters_input->use_tvb_limiter;


    if (nstate_input == 1) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 1, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 1, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 1, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 1, real> >(parameters_input);
        }
    }
    else if (nstate_input == 2) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 2, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 2, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 2, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 2, real> >(parameters_input);
        }
    }
    else if (nstate_input == 3) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 3, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 3, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 3, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 3, real> >(parameters_input);
        }
    }
    else if (nstate_input == 4) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 4, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 4, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 4, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 4, real> >(parameters_input);
        }
    }
    else if (nstate_input == 5) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 5, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 5, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 5, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 5, real> >(parameters_input);
        }
    }
    else if (nstate_input == 6) {
        if (limiter_type == limiter_enum::none)
        {
            if (apply_tvb == true)
                return std::make_shared < TVBLimiter<dim, 6, real> >(parameters_input);
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle)
        {
            return std::make_shared< MaximumPrincipleLimiter<dim, 6, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2010)
        {
            return std::make_shared< PositivityPreservingLimiter<dim, 6, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preserving2011)
        {
            return std::make_shared< PositivityPreservingLimiterRobust<dim, 6, real> >(parameters_input);
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
