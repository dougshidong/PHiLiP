#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Models Base Class
//================================================================
template <int dim, int nstate, typename real>
ModelBase<dim, nstate, real>::ModelBase(
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input):
        manufactured_solution_function(manufactured_solution_function_input)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
ModelBase<dim,nstate,real>::~ModelBase() {}
//----------------------------------------------------------------

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelBase<PHILIP_DIM, 1, double>;
template class ModelBase<PHILIP_DIM, 2, double>;
template class ModelBase<PHILIP_DIM, 3, double>;
template class ModelBase<PHILIP_DIM, 4, double>;
template class ModelBase<PHILIP_DIM, 5, double>;
template class ModelBase<PHILIP_DIM, 8, double>;

template class ModelBase<PHILIP_DIM, 1, FadType>;
template class ModelBase<PHILIP_DIM, 2, FadType>;
template class ModelBase<PHILIP_DIM, 3, FadType>;
template class ModelBase<PHILIP_DIM, 4, FadType>;
template class ModelBase<PHILIP_DIM, 5, FadType>;
template class ModelBase<PHILIP_DIM, 8, FadType>;

template class ModelBase<PHILIP_DIM, 1, RadType>;
template class ModelBase<PHILIP_DIM, 2, RadType>;
template class ModelBase<PHILIP_DIM, 3, RadType>;
template class ModelBase<PHILIP_DIM, 4, RadType>;
template class ModelBase<PHILIP_DIM, 5, RadType>;
template class ModelBase<PHILIP_DIM, 8, RadType>;

template class ModelBase<PHILIP_DIM, 1, FadFadType>;
template class ModelBase<PHILIP_DIM, 2, FadFadType>;
template class ModelBase<PHILIP_DIM, 3, FadFadType>;
template class ModelBase<PHILIP_DIM, 4, FadFadType>;
template class ModelBase<PHILIP_DIM, 5, FadFadType>;
template class ModelBase<PHILIP_DIM, 8, FadFadType>;

template class ModelBase<PHILIP_DIM, 1, RadFadType>;
template class ModelBase<PHILIP_DIM, 2, RadFadType>;
template class ModelBase<PHILIP_DIM, 3, RadFadType>;
template class ModelBase<PHILIP_DIM, 4, RadFadType>;
template class ModelBase<PHILIP_DIM, 5, RadFadType>;
template class ModelBase<PHILIP_DIM, 8, RadFadType>;

} // Physics namespace
} // PHiLiP namespace