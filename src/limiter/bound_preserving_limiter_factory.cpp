#include "parameters/all_parameters.h"
#include "bound_preserving_limiter_factory.hpp"
#include "bound_preserving_limiter.h"
#include "tvb_limiter.h"
#include "maximum_principle_limiter.h"
#include "positivity_preserving_limiter.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
std::unique_ptr< BoundPreservingLimiter<dim, real> >
    BoundPreservingLimiterFactory<dim, nstate, real>
    ::create_limiter(
        const Parameters::AllParameters* const parameters_input)
{
    if (nstate == parameters_input->nstate)
        return BoundPreservingLimiterFactory<dim, nstate, real>::select_limiter(parameters_input);
    else if constexpr (nstate > 1)
        return BoundPreservingLimiterFactory<dim, nstate - 1, real>::create_limiter(parameters_input);
    else
        return nullptr;
}

template <int dim, int nstate, typename real>
std::unique_ptr< BoundPreservingLimiter<dim, real> >
    BoundPreservingLimiterFactory<dim, nstate, real>
    ::select_limiter(
        const Parameters::AllParameters* const parameters_input)
{
    using limiter_enum = Parameters::LimiterParam::LimiterType;
    using flux_nodes_enum = Parameters::AllParameters::FluxNodes;

    limiter_enum limiter_type = parameters_input->limiter_param.bound_preserving_limiter;
    flux_nodes_enum flux_nodes_type = parameters_input->flux_nodes_type;

    bool apply_tvb = parameters_input->limiter_param.use_tvb_limiter;
    bool curvilinear_grid = parameters_input->use_curvilinear_grid;

    if (limiter_type == limiter_enum::none) {
        if (apply_tvb == true) {
            if(curvilinear_grid) {
                std::cout << "Error: Cannot create limiter for curvilinear grid" << std::endl;
                std::abort();
            } else if (flux_nodes_type != flux_nodes_enum::GLL) {
                std::cout << "Error: Can only use limiter with GLL flux nodes" << std::endl;
                std::abort();
            } else if (dim == 1)
                return std::make_unique < TVBLimiter<dim, nstate, real> >(parameters_input);
            else {
                std::cout << "Error: Cannot create TVB limiter for dim > 1" << std::endl;
                std::abort();
            }
        }
        else
            return nullptr;
    } else if(curvilinear_grid) {
        std::cout << "Error: Cannot create limiter for curvilinear grid" << std::endl;
        std::abort();
    } else if (flux_nodes_type != flux_nodes_enum::GLL) {
        std::cout << "Error: Can only use limiter with GLL flux nodes" << std::endl;
        std::abort();
    } else if (limiter_type == limiter_enum::maximum_principle) {
        return std::make_unique< MaximumPrincipleLimiter<dim, nstate, real> >(parameters_input);
    } else if (limiter_type == limiter_enum::positivity_preservingZhang2010
                || limiter_type == limiter_enum::positivity_preservingWang2012) {
        if (nstate == dim + 2)
            return std::make_unique< PositivityPreservingLimiter<dim, nstate, real> >(parameters_input);
        else {
            if(nstate != dim + 2) {
                std::cout << "Error: Cannot create Positivity-Preserving limiter for nstate_input != dim + 2" << std::endl;
                std::abort();
            }
        }
    }

    std::cout << "Error: Cannot create limiter pointer due to an invalid limiter type specified" << std::endl;
    std::abort();
    return nullptr;
}

    template class BoundPreservingLimiterFactory <PHILIP_DIM, 1, double>;
    template class BoundPreservingLimiterFactory <PHILIP_DIM, 2, double>;
    template class BoundPreservingLimiterFactory <PHILIP_DIM, 3, double>;
    template class BoundPreservingLimiterFactory <PHILIP_DIM, 4, double>;
    template class BoundPreservingLimiterFactory <PHILIP_DIM, 5, double>;
    template class BoundPreservingLimiterFactory <PHILIP_DIM, 6, double>;
} // PHiLiP namespace