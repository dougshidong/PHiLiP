#ifndef __LINEAR_SOLVER_H__
#define __LINEAR_SOLVER_H__

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "parameters/all_parameters.h"

namespace PHiLiP {

    /// Still need to make a LinearSolver class for our problems
    /// Note that right hand side should be const
    /// however, the Trilinos wrapper gives and error when trying to
    /// map it. This is probably because the Trilinos function 
    /// does not take right_hand_side as a const
    std::pair<unsigned int, double>
        solve_linear ( dealii::TrilinosWrappers::SparseMatrix &system_matrix,
                       dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
                       dealii::LinearAlgebra::distributed::Vector<double> &solution,
                       const Parameters::LinearSolverParam &param);

    std::pair<unsigned int, double>
    solve_linear ( const dealii::TrilinosWrappers::SparseMatrix &system_matrix,
                   const dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
                   dealii::LinearAlgebra::distributed::Vector<double> &solution,
                   const Parameters::LinearSolverParam &param);

} // PHiLiP namespace

#endif
