/** From Eigen NNLS:
 * Non-Negagive Least Squares Algorithm for Eigen.
 *
 * Copyright (C) 2021 Essex Edwards, <essex.edwards@gmail.com>
 * Copyright (C) 2013 Hannes Matuschek, hannes.matuschek at uni-potsdam.de
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * 
 * Non-Negative Least Squares (NNLS) Module
 * This module provides a single class @c Eigen::NNLS implementing the NNLS algorithm.
 * The algorithm is described in "SOLVING LEAST SQUARES PROBLEMS", by Charles L. Lawson and
 * Richard J. Hanson, Prentice-Hall, 1974 https://epubs.siam.org/doi/10.1137/1.9781611971217
 * and solves optimization problems of the form
 *
 * \f[ \min \left\Vert Ax-b\right\Vert_2^2\quad s.t.\, x\ge 0\,.\f]
 *
 * The algorithm solves the constrained least-squares problem above by iteratively improving
 * an estimate of which constraints are active (elements of \f$x\f$ equal to zero)
 * and which constraints are inactive (elements of \f$x\f$ greater than zero).
 * Each iteration, an unconstrained linear least-squares problem solves for the
 * components of \f$x\f$ in the (estimated) inactive set and the sets are updated.
 * The unconstrained problem minimizes \f$\left\Vert A^Nx^N-b\right\Vert_2^2\f$,
 * where \f$A^N\f$ is a matrix formed by selecting all columns of A which are
 * in the inactive set \f$N\f$.
 *
 * The Eigen NNLS was updated to instead use Epetra Structures and linear solvers to work more 
 * conducively with the PHiLiP and more specifically the reduced order modelling classes.
 */

/* Note: This solver is based on a modified version of the algorithm descrived in "SOLVING 
 * LEAST SQUARES PROBLEMS". This is because in the context of Energy Conserving Sampling
 * and Weighting a modified exit condition was introduced. The details regarding the exit
 * condition (EQUATION 13) are described in the following paper:
 * https://onlinelibrary.wiley.com/doi/full/10.1002/nme.5332
 * More details on the ECSW hyper-reduction technique can be found in:
 * https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4820
*/

#ifndef NNLS_H
#define NNLS_H

#include <iostream>
#include <string>
#include <AztecOO_config.h>
#include <Epetra_MpiComm.h>
#include <Epetra_ConfigDefs.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Import.h>
#include <Epetra_LinearProblem.h>
#include <EpetraExt_MatrixMatrix.h>
#include <AztecOO.h>
#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include <eigen/Eigen/Dense>
#include "parameters/all_parameters.h"

using Eigen::Matrix;
namespace PHiLiP {
/**  Non-Negagive Least Squares Solver for Epetra Structures
 *   Based on the NNLS in Eigen unsupported and in MATLAB:
 *   https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/Eigen/NNLS
 *   https://www.mathworks.com/help/matlab/ref/lsqnonneg.html
 */

class NNLS_solver
{
public:
    /// Default Constructor
    NNLS_solver(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const Epetra_CrsMatrix &A, 
        Epetra_MpiComm &Comm, 
        Epetra_Vector &b);

    /// Constructor w/ Gradient Exit Condition
    NNLS_solver(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const Epetra_CrsMatrix &A, 
        Epetra_MpiComm &Comm, 
        Epetra_Vector &b, 
        bool grad_exit_crit);
    
    /// Constructor w/ Iterative Linear Solver
    NNLS_solver(        
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const Epetra_CrsMatrix &A, 
        Epetra_MpiComm &Comm, 
        Epetra_Vector &b, 
        bool iter_solver, 
        int LS_iter, 
        double LS_tol);

    /// Common Constructor w/ Gradient Exit Condition & Iterative Linear Solver
    NNLS_solver(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const Epetra_CrsMatrix &A, 
        Epetra_MpiComm &Comm, 
        Epetra_Vector &b, 
        bool grad_exit_crit, 
        bool iter_solver, 
        int LS_iter, 
        double LS_tol);
 
    /// Destructor
    virtual ~NNLS_solver() {};

    /// Call to solve NNLS problem
    bool solve();

    /// Returns protected approximate solution
    Epetra_Vector & getSolution() {return multi_x_;}

    /// Initiliazes the solution vector, must be used before .solve is called
    void startingSolution(Epetra_Vector &start) {
        Epetra_Vector start_single_core = allocateVectorToSingleCore(start);
        P.flip();
        SubIntoX(start_single_core);
        P.flip();}
  
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

protected:
    /// Epetra Commuicator Object with MPI
    Epetra_MpiComm Comm_;
    /// Epetra Matrix A allocated to a single core
    const Epetra_CrsMatrix A_;
    /// Epetra Vector b allocated to a single core
    Epetra_Vector b_;
    /// Epetra Vector x, to be solved. Allocated to a single core
    Epetra_Vector x_;
    /// Epetra_Vector x, to be solved. Allocated to multiple cores
    Epetra_Vector multi_x_;
    /// Needed if the an iterative solver is used
    int LS_iter_;
    /// Needed if the an iterative solver is used
    double LS_tol_; 
    /// Number of inactive points
    int numInactive_;
    /// Vector of booleans representing the columns in the active set
    std::vector<bool> Z;
    /// Vector of boolean representing the columns in the inactive set
    std::vector<bool> P;

public:
    /// Boolean used for iterative solvers
    bool iter_solver_;
    /// Boolean to use an exit Condition depending on maximum gradent
    bool grad_exit_crit_;
    /// Number of iterations in the NNLS solver
    int iter_;

protected:
    /// Eigen Vector of the index_set
    Eigen::VectorXd index_set;

private:
    /// Creates square permutation matrix based off the active/inactive set, no longer in use
    void Epetra_PermutationMatrix(Epetra_CrsMatrix &P_mat);

    /// Creates a matrix using the columns in A in the set P
    void PositiveSetMatrix(Epetra_CrsMatrix &P_mat);

    /// Replaces the entries with x with the values in temp
    void SubIntoX(Epetra_Vector &temp);

    /// Adds the values of temp times alpha into the solution vector x 
    void AddIntoX(Epetra_Vector &temp, double alpha);

    /// Moves the column at idx into the active set (updating the index_set, Z, P, and numInactive_)
    void moveToActiveSet(int idx);

    /// Moves the column at idx into the inactive set (updating the index_set, Z, P, and numInactive_)
    void moveToInactiveSet(int idx);

    /** @brief Re-allocates solution to multiple cores
    *   @param c Epetra_vector to reallocate on multiple cores
    *   @return An Epetra_vector with the data of c distributed on multiple cores
    */
    Epetra_Vector allocateVectorToMultipleCores(const Epetra_Vector &c);

    /** @brief Allocates the incoming A Matrix to a single core
    *   @param A Epetra_CrsMatrix to reallocate on a single core
    *   @return An Epetra_CrsMatrix with the data of A distributed on a single core
    */
    Epetra_CrsMatrix allocateMatrixToSingleCore(const Epetra_CrsMatrix &A);
    
    /** @brief Allocates the incoming b vector to a single core
    *   @param b Epetra_Vector to reallocate on a single core
    *   @return An Epetra_Vector with the data of b distributed on a single core
    */
    Epetra_Vector allocateVectorToSingleCore(const Epetra_Vector &b);
};
} // PHiLiP namespace
#endif  // NNLS_H
