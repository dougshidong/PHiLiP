#include "full_order_solution.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
FOMSolution<dim, nstate>::FOMSolution(std::shared_ptr<DGBase<dim,double>> &dg_input, Functional<dim,nstate,double> &functional_input, double sensitivity)
        : state(dg_input->solution)
        , sensitivity(sensitivity)
        , functional_value(functional_input.evaluate_functional(false, false, false))
{
}

template class FOMSolution <PHILIP_DIM, 1>;
template class FOMSolution <PHILIP_DIM, 2>;
template class FOMSolution <PHILIP_DIM, 3>;
template class FOMSolution <PHILIP_DIM, 4>;
template class FOMSolution <PHILIP_DIM, 5>;

}
}