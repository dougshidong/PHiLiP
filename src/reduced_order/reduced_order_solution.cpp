#include "reduced_order_solution.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
ROMSolution<dim, nstate>::ROMSolution(std::shared_ptr<DGBase<dim,double>> &dg_input, std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix_transpose, Functional<dim,nstate,double> &functional_input, std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis)
        : system_matrix_transpose(system_matrix_transpose)
        , right_hand_side(dg_input->right_hand_side)
        , basis(pod_basis)
        , functional_value(functional_input.evaluate_functional( true, false, false))
        , gradient(functional_input.dIdw)
{
}

template class ROMSolution <PHILIP_DIM, 1>;
template class ROMSolution <PHILIP_DIM, 2>;
template class ROMSolution <PHILIP_DIM, 3>;
template class ROMSolution <PHILIP_DIM, 4>;
template class ROMSolution <PHILIP_DIM, 5>;

}
}