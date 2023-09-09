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
    using limiter_enum = Parameters::LimiterParam::LimiterType;
    limiter_enum limiter_type = parameters_input->limiter_param.bound_preserving_limiter;
    bool apply_tvb = parameters_input->limiter_param.use_tvb_limiter;

    if (nstate_input == 1) {
        if (limiter_type == limiter_enum::none) {
            if (apply_tvb == true) {
                if (dim == 1)
                    return std::make_shared < TVBLimiter<dim, 1, real> >(parameters_input);
                else {
                    assert(0 == 1 && "Cannot create TVB limiter for dim > 1");
                    return nullptr;
                }
            }
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 1, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if(nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 1, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 1, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
    }
    else if (nstate_input == 2) {
        if (limiter_type == limiter_enum::none) {
            if (apply_tvb == true) {
                if (dim == 1)
                    return std::make_shared < TVBLimiter<dim, 2, real> >(parameters_input);
                else {
                    assert(0 == 1 && "Cannot create TVB limiter for dim > 1");
                    return nullptr;
                }
            }
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 2, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 2, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 2, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
    }
    else if (nstate_input == 3) {
        if (limiter_type == limiter_enum::none) {
            if (apply_tvb == true) {
                if (dim == 1)
                    return std::make_shared < TVBLimiter<dim, 3, real> >(parameters_input);
                else {
                    assert(0 == 1 && "Cannot create TVB limiter for dim > 1");
                    return nullptr;
                }
            }
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 3, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 3, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 3, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
    }
    else if (nstate_input == 4) {
        std::cout << "nstate_input is 4" << std::endl;
        if (limiter_type == limiter_enum::none) {
            std::cout << "limiter type is none" << std::endl;
            if (apply_tvb == true) {
                std::cout << "apply_tvb is true" << std::endl;
                if (dim == 1)
                    return std::make_shared < TVBLimiter<dim, 4, real> >(parameters_input);
                else {
                    std::cout << "dim > 1" << std::endl;
                    assert(0 == 1 && "Cannot create TVB limiter for dim > 1");
                    return nullptr;
                }
            }
            else
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 4, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 4, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 4, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
    }
    else if (nstate_input == 5) {
        if (limiter_type == limiter_enum::none) {
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 5, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 5, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 5, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
    }
    else if (nstate_input == 6) {
        if (limiter_type == limiter_enum::none) {
                return nullptr;
        }
        else if (limiter_type == limiter_enum::maximum_principle) {
            return std::make_shared< MaximumPrincipleLimiter<dim, 6, real> >(parameters_input);
        }
        else if (limiter_type == limiter_enum::positivity_preservingZhang2010) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Zhang2010<dim, 6, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
            }
        }
        else if (limiter_type == limiter_enum::positivity_preservingWang2012) {
            if (nstate_input == dim + 2)
                return std::make_shared< PositivityPreservingLimiter_Wang2012<dim, 6, real> >(parameters_input);
            else {
                assert(0 == 1 && "Cannot create Positivity-Preserving limiter for nstate_input != dim + 2");
                return nullptr;
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
