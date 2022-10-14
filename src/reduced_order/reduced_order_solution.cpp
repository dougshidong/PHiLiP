#include "reduced_order_solution.h"

#include <utility>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
ROMSolution<dim, nstate>::ROMSolution(Parameters::AllParameters params, dealii::LinearAlgebra::distributed::Vector<double> _solution, dealii::LinearAlgebra::distributed::Vector<double> _gradient)
        : params(params)
        , solution(_solution)
        , gradient(_gradient)
{
}

template class ROMSolution <PHILIP_DIM, 1>;
template class ROMSolution <PHILIP_DIM, 2>;
template class ROMSolution <PHILIP_DIM, 3>;
template class ROMSolution <PHILIP_DIM, 4>;
template class ROMSolution <PHILIP_DIM, 5>;

}
}