#include <deal.II/base/geometry_info.h>
#include "bound_preserving_limiter_factory.hpp"
#include "bound_preserving_limiter.h"

namespace PHiLiP {
namespace LIMITER {

template <int dim, typename real>
std::shared_ptr< BoundPreservingLimiterBase<dim,real> >//returns type OperatorsBase
BoundPreservingLimiterFactory<dim,real>
::create_operators(
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input)
{
        if(nstate_input == 1){
            return std::make_shared< BoundPreservingLimiterBase<dim,real,1> >(parameters_input);//
        }
        else if(nstate_input == 2){
            return std::make_shared< BoundPreservingLimiterBase<dim,real,2> >(parameters_input);//
        }
        else if(nstate_input == 3){
            return std::make_shared< BoundPreservingLimiterBase<dim,real,3> >(parameters_input);//
        }
        else if(nstate_input == 4){
            return std::make_shared< BoundPreservingLimiterBase<dim,real,4> >(parameters_input);//
        }
        else if(nstate_input == 5){
            return std::make_shared< BoundPreservingLimiterBase<dim,real,5> >(parameters_input);//
        }
        else if (nstate_input == 6) {
            return std::make_shared< BoundPreservingLimiterBase<dim,real,6> >(parameters_input);//
        }
        else{
            std::cout<<"Number of states "<<nstate_input<<"not supported."<<std::endl;
            return nullptr;
        }
}

template class BoundPreservingLimiterFactory <PHILIP_DIM, double>;

} // LIMITER namespace
} // PHiLiP namespace
