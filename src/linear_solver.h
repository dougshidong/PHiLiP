#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace PHiLiP
{
    using namespace dealii;

    template <typename real>
    std::pair<unsigned int, double>
    //LinearSolver<real>::solve_linear (
    solve_linear (
        const TrilinosWrappers::SparseMatrix &system_matrix,
        Vector<real> &right_hand_side, 
        Vector<real> &solution);
}
