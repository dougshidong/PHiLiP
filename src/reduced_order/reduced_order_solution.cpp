#include "reduced_order_solution.h"

#include <utility>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nspecies, int nstate>
ROMSolution<dim, nspecies, nstate>::ROMSolution(Parameters::AllParameters params, dealii::LinearAlgebra::distributed::Vector<double> _solution, dealii::LinearAlgebra::distributed::Vector<double> _gradient)
        : params(params)
        , solution(_solution)
        , gradient(_gradient)
{
}

#if PHILIP_SPECIES==1
template class ROMSolution <PHILIP_DIM, PHILIP_SPECIES, 1>;
template class ROMSolution <PHILIP_DIM, PHILIP_SPECIES, 2>;
template class ROMSolution <PHILIP_DIM, PHILIP_SPECIES, 3>;
template class ROMSolution <PHILIP_DIM, PHILIP_SPECIES, 4>;
template class ROMSolution <PHILIP_DIM, PHILIP_SPECIES, 5>;
#endif
}
}