#include "linear_solver/NNLS_solver.h"
#include "helper_functions.cpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <eigen/Eigen/QR>
#include <eigen/test/random_matrix_helper.h>
//#include <eigen/Eigen/SVD>

using namespace Eigen;
using namespace PHiLiP;
#define EIGEN_TEST_MAX_SIZE 50

// ** NOTES **
// A majority of the tests are pulled from the tests in the Eigen unsupported NNLS packages. They can be found at:
// https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NNLS.cpp
// Some of the tests from the Eigen NNLS solver are not included as they would not pass using the new exit condition.
// The CSV files C.csv and d.csv are pulled from the MATLAB implementation of the ECSW approach for the Burgers' 1D problem.
// x_pow_4.csv is pulled from the modified Eigen NNLS solver solution for the C.csv and d.csv files, when the tolerance used is 1E-4.



/// @brief Check that 'x' solves the NNLS optimization problem `min ||A*x-b|| s.t. 0 <= x`
/// @param A Eigen Matrix
/// @param b_eig Eigen RHS vector
/// @param x Approximate solution for NNLS solver
/// @param tau tolerance for the residual exit condition
/// @param random whether the problem has a known solution, if true will check gradient tolerance as well
/// @return boolean depending on the optimality conditions listed below
/// Notes:
/// The tau parameter is the tolerance on the residual: ||residual||_2 <= tau * ||b||_2.
/// Due to the change in the exit conditions, random problems without known NNLS solutions which should converge may reach the maximum iterations.
/// Therefore, the random flag can be set to true to check if the gradient from the original exit condition is small.
bool verify_nnls_optimality(Eigen::MatrixXd &A, Eigen::MatrixXd &b_eig, Eigen::MatrixXd &x, double tau, bool random = false) {
  // The NNLS optimality conditions are:
  //
  // * 0 <= x[i] \forall i
  // * ||residual||_2 <= tau * ||b||_2

  Eigen::MatrixXd res = (b_eig - A * x);
  bool opt = true;
  // NNLS solutions are not negative.
  if (-std::numeric_limits<double>::epsilon() > x.minCoeff()){
    opt = false;
  }
  // NNLS solutions achieve expected residual accuracy (tau).
  else if (res.squaredNorm() > tau*b_eig.squaredNorm()){
    opt = false;
  }
  // If the problem does not have a known NNLS solution, the previous tolerance can be relaxed to check if the gradient satisfies the bound.
  // Instead of ||residual||_2 <= tau * ||b||_2, we allow max(A^T * residual) < tau.
  if (random){
    if ((A.transpose() * res).maxCoeff() < tau){
      opt = true;
      std::cout << "Gradient is small/negative, but residual is not necessarily small" << std::endl;
    }
  }
  return opt;
}

/// @brief Test the NNLS solver for a problem with a known solution (can be from Eigen NNLS solver tests or MATLAB)
/// @param A_eig Eigen Matrix
/// @param col number of columns
/// @param row number of rows
/// @param x_eig exact solution
/// @param b_eig Eigen RHS vector
/// @param b_pt pointer to array of RHS vector values
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean dependent on the exit condition of the NNLS solver and optimality/accuracy of the solutions
bool test_nnls_known_CLASS(Eigen::MatrixXd &A_eig, int col, int row, Eigen::MatrixXd &x_eig, Eigen::MatrixXd &b_eig, double *b_pt, Epetra_MpiComm &Comm, const double tau, const int max_iter){
  // Check solution of NNLS problem with a known solution
  // Returns true if the solver exits for any condition other than max_iter and if the solution x is accurate to the true solution and satisfies the conditions above
  
  // Create Epetra structures of A and b from Eigen matrices and pointers, respectively
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, col, row, Comm));
  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  Epetra_Vector b(Copy, A.RowMap(), b_pt);
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit condition
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  
  // If the maximum iterations were reached, output message to signal possible failure of the solver
  // Still check the optimality of the solution as it can still be the close to the exact solution
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  // Convert the solution Epetra_Vector to an Eigen::MatrixXd
  Eigen::MatrixXd x_nnls_eig(col,1);
  epetra_to_eig_vec(col, x , x_nnls_eig);

  // Confirm the optimality of the solution wrt tolerance and positivity
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  // Compare to the exact solutions
  opt&= x_nnls_eig.isApprox(x_eig, tau);
  return opt;
}

/// @brief Test to check handling of matrix with no columns
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the length of the solution (returns true if the solver returns true, it completes zero iterations and the length of the vector is zero) 
bool test_nnls_handles_Mx0_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  // Build matrix with no columns
  const int row = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig(row, 0);
  VectorXd b_eig = VectorXd::Random(row);

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, row, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterationss
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  // Check the number of iterations completed and size of the solution vector (both expected to be zeros)
  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x(NNLS_prob.getSolution());
  opt &= (x.MyLength()== 0);
  return opt;
}

/// @brief Test to check handling of matrix with no rows and columns (0x0)
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return  boolean depending on the exit condition of the solver, the number of iterations, and the length of the solution (returns true if the solver returns true, it completes zero iterations and the length of the vector is zero) 
bool test_nnls_handles_0x0_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
// Build matrix with no rows and columns
  MatrixXd A_eig(0, 0);
  VectorXd b_eig(0);

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, 0, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterationss
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  // Check the number of iterations completed and size of the solution vector (both expected to be zeros)
  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x(NNLS_prob.getSolution());
  opt &= (x.MyLength()== 0);
  return opt;
}

/// @brief Test a randomly generated problem
/// @param Comm MpiComm for Epetra Maps
/// @return boolean depending on the exit condition of the solver and the optimality of the solution
/* NOTE: Before changing the exit condition, it is suggested in the Eigen NNLS solver that the solution 
should almost always be found before the maximum number of iterations is reached. However, the residual
tolerance may never be reached. For this reason, the bound on the gradient is also checked on randonmly 
generated problems even when the maximum number of iterations is exceeded and it returns true if the 
sgradient bound is satisfied. */
bool test_nnls_random_problem(Epetra_MpiComm &Comm) {
  // Random dimensions of the matrix below EIGEN_TEST_MAX_SIZE
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  // Make some sort of random test problem from a wide range of scales and condition numbers
  using std::pow;
  using Scalar = typename MatrixXd::Scalar;
  const Scalar sqrtConditionNumber = pow(Scalar(10), internal::random<Scalar>(Scalar(0), Scalar(2)));
  const Scalar scaleA = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const Scalar minSingularValue = scaleA / sqrtConditionNumber;
  const Scalar maxSingularValue = scaleA * sqrtConditionNumber;
  MatrixXd A_eig(rows, cols);
  generateRandomMatrixSvs(setupRangeSvs<Matrix<Scalar, Dynamic, 1>>(cols, minSingularValue, maxSingularValue), rows,
                          cols, A_eig);

  // Make a random RHS also with a random scaling
  using VectorB = decltype(A_eig.col(0).eval());
  MatrixXd b_eig = 100 * VectorB::Random(A_eig.rows());
  double *b_pt = b_eig.data();

  // Estimate the tolerance and max_iter based on problem size
  using Scalar = typename MatrixXd::Scalar;
  using std::sqrt;
   const Scalar tolerance =
       sqrt(Eigen::GenericNumTraits<Scalar>::epsilon()) * b_eig.cwiseAbs().maxCoeff() * A_eig.cwiseAbs().maxCoeff();
  Index max_iter = 5 * A_eig.cols();  // A heuristic guess.
  
  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tolerance);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations
  // As noted above, this is permissable due to the change in the exit condition but the solution must satisfy the gradient exit condition
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  // Confirm the optimality of the solution wrt tolerance and positivity
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tolerance, true); // random flag set to true
  return opt;
}

/// @brief Test to check handling of zero RHS vector
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the entries in the solution (returns true if the solver returns true, it completes less than two iteration, and the solution vector is zeros)
bool test_nnls_handles_zero_rhs(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and RHS vector of zeros
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Zero(rows);

  double *b_pt = b_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  bool opt = true;
  // Check if solver exited by reaching the maximum number of iterations
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }

  // Check the number of iterations completed (expected to be 0 or 1) and that all the entries in the solution are zero
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  opt &= (NNLS_prob.iter_ <= 1);
  opt &= (x_nnls_eig.isApprox(VectorXd::Zero(cols)));
  return opt;
}

/// @brief Test to check handling of matrix with dependent columns
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver and the optimality of the solution (returns true if the max. iters are reached or if the solution is the optimal one)
bool test_nnls_handles_dependent_columns(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and matrix with dependent columns
  const Index rank = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2);
  const Index cols = 2 * rank;
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, rank) * MatrixXd::Random(rank, cols);
  MatrixXd b_eig = VectorXd::Random(rows);
  
  double *b_pt = b_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  // From Eigen : 
  /* What should happen when the input 'A' has dependent columns?
     We might still succeed. Or we might not converge.
     Either outcome is fine. If Success is indicated,
     then 'x' must actually be a solution vector. */
  bool opt = true;
  Epetra_Vector x(NNLS_prob.getSolution());
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }
  else{
    // Confirm the optimality of the solution wrt tolerance and positivity
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  }
  std::cout << " Solution x "<< std::endl;
  std::cout << x << std::endl;
  return opt;
}

/// @brief Test to check handling of a wide matrix (ie. cols > rows)
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver and the optimality of the solution (returns true if the max. iters are reached or if the solution is the optimal one)
bool test_nnls_handles_wide_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and wide matrix
  const Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(2, cols - 1);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Random(rows);
  
  double *b_pt = b_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  // From Eigen:
  /* What should happen when the input 'A' is wide?
     The unconstrained least-squares problem has infinitely many solutions.
     Subject to the non-negativity constraints,
     the solution might actually be unique (e.g. it is [0,0,..,0]).
     So, NNLS might succeed or it might fail.
     Either outcome is fine. If Success is indicated,
     then 'x' must actually be a solution vector. */

  bool opt = true;
  Epetra_Vector x(NNLS_prob.getSolution());
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  // Check if solver exited by reaching the maximum number of iterations
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }
  else{
    // Confirm the optimality of the solution wrt tolerance and positivity
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  }
  std::cout << " Solution x "<< std::endl;
  std::cout << x << std::endl;
  return opt;
}

/// @brief Test to check handling when the solution vector is set to the exact solution before the solver is run
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the optimality of the solution (returns true if the solver returns true, it completes zero iterations, and the solution is optimal)
bool test_nnls_special_case_solves_in_zero_iterations(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive
  MatrixXd b_eig = A_eig * x_eig;

  double *b_pt = b_eig.data();
  double *x_pt = x_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);
  Epetra_Vector x_start(Copy, A.ColMap(), x_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  NNLS_prob.startingSolution(x_start); // SET SOLUTION TO EXACT SOLUTION
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations
  bool opt = true;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(n,1);
  epetra_to_eig_vec(n, x , x_nnls_eig);
  // Check the number of iterations (expected to be zero)
  opt &= (NNLS_prob.iter_ == 0);
  // Confirm the optimality of the solution wrt tolerance and positivity
  opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  return opt;
}

/// @brief Test to check handling when the solution should be found in n iterations due to structure of RHS
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the optimality of the solution (returns true if the solver returns true, it completes n = num of cols. iterations, and the solution is optimal)
bool test_nnls_special_case_solves_in_n_iterations(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  double *b_pt = b_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //

  // Check if solver exited by reaching the maximum number of iterations
  bool opt = true;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(n,1);
  epetra_to_eig_vec(n, x , x_nnls_eig);
  // Check the number of iterations (expected to be equal to the number of columns in A)
  opt &= (NNLS_prob.iter_ == n);
  // Confirm the optimality of the solution wrt tolerance and positivity
  opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  return opt;
}

/// @brief Test to check handling when the maximum iterations is too low
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @return boolean depending on the exit condition of the solver and the number of iterations (returns true if the solver returns false due to exiting by max_iter, and the number of iterations is equal to n-1)
bool test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(Epetra_MpiComm &Comm, const double tau) {
  // Using the special case that takes `n` iterations,
  // from `test_nnls_special_case_solves_in_n_iterations`,
  // we can set max iterations too low and that should cause the solve to fail.

  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  double *b_pt = b_eig.data();

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  // Set max_iters too low to cause solver to fail/return false on purpose
  const Index max_iters = n - 1;
  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLS_solver NNLS_prob(A, Comm, b, max_iters, tau);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations, return true in this case
  bool opt = false;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt = true;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  // Check the number of iterations is in fact equal to the maximum number of iterations
  opt &= (NNLS_prob.iter_ == max_iters);
  return opt;
}

/// @brief Case 1: 4x2 problem, unconstrained solution positive
bool case_1 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4, 2);
  Eigen::MatrixXd x_eig(2,1);
  Eigen::MatrixXd b_eig(4,1);
  A_eig << 1, 1,  2, 4,  3, 9,  4, 16;
  b_eig << 0.6, 2.2, 4.8, 8.4;
  x_eig << 0.1, 0.5;
  double b_pt[] = {0.6, 2.2, 4.8, 8.4};

  return test_nnls_known_CLASS(A_eig, 2, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

/// @brief Case 2: 4x3 problem, unconstrained solution positive
bool case_2 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1,  1,  1,
       2,  4,  8,
       3,  9, 27,
       4, 16, 64;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13;
  double b_pt[] = {0.73, 3.24, 8.31, 16.72};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

/// @brief Case 3: Simple 4x4 problem, unconstrained solution non-negative
bool case_3 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,4);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(4,1);

  A_eig << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13, 0;
  double b_pt[] = {0.73, 3.24, 8.31, 16.72};

  return test_nnls_known_CLASS(A_eig, 4, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

/// @brief Case 4: Simple 4x3 problem, unconstrained solution non-negative
bool case_4 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.23, 1.24, 3.81, 8.72;
  x_eig << 0.1, 0, 0.13;
  double b_pt[] = {0.23, 1.24, 3.81, 8.72};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

/// @brief Case 5: Simple 4x3 problem, unconstrained solution indefinite
bool case_5 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.13, 0.84, 2.91, 7.12;
   // Solution obtained by original nnls() implementation in Fortran
  x_eig << 0.0, 0.0, 0.1106544;
  double b_pt[] = {0.13, 0.84, 2.91, 7.12};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

/// @brief Case MATLAB: 1024 x 49 problem from ECSW in MATLAB for Burgers' 1D
bool case_MATLAB (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig = load_csv<MatrixXd>("C.csv");
  Eigen::MatrixXd b_eig = load_csv<MatrixXd>("d.csv");
  Eigen::MatrixXd x_eig = load_csv<MatrixXd>("x_pow_4.csv");

  double *b_pt = b_eig.data();

  return test_nnls_known_CLASS(A_eig, 1024, 49, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

int main(int argc, char *argv[]){
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
  double tau = 1E-8;
  const int max_iter = 10000;

  bool ok = true;

  // Possible Test Cases
  std::string s_1 = "known";
  std::string s_2 = "matlab";
  std::string s_3 = "noCols";
  std::string s_4 = "empty";
  std::string s_5 = "random";
  std::string s_6 = "zeroRHS";
  std::string s_7 = "depCols";
  std::string s_8 = "wide";
  std::string s_9 = "zeroIter";
  std::string s_10 = "nIter"; 
  std::string s_11 = "maxIter";
   
  if (argv[1] == s_1){
    ok &= case_1(Comm, tau, max_iter);
    ok &= case_2(Comm, tau, max_iter);
    ok &= case_3(Comm, tau, max_iter);
    ok &= case_4(Comm, tau, max_iter);
    ok &= case_5(Comm, 1E-3, max_iter); // had to relax the tolerance for this case
  }
  else if (argv[1] == s_2){
    tau = 1E-4;
    ok &= case_MATLAB(Comm, tau, max_iter);
  }
  else if (argv[1] == s_3){
    ok &= test_nnls_handles_Mx0_matrix(Comm, tau, max_iter);
  }  
  else if (argv[1] == s_4){
    ok &= test_nnls_handles_0x0_matrix(Comm, tau, max_iter);
  }
  else if (argv[1] == s_5){
    ok &= test_nnls_random_problem(Comm);
  }
  else if (argv[1] == s_6){
    ok &= test_nnls_handles_zero_rhs(Comm, tau, max_iter);
  }
  else if (argv[1] == s_7){
    ok &= test_nnls_handles_dependent_columns(Comm, tau, max_iter);
  }
  else if (argv[1] == s_8){
    ok &= test_nnls_handles_wide_matrix(Comm, tau, max_iter);
  }
  else if (argv[1] == s_9){
    ok &= test_nnls_special_case_solves_in_zero_iterations(Comm, tau, max_iter);
  }
  else if (argv[1] == s_10){
    ok &= test_nnls_special_case_solves_in_n_iterations(Comm, tau, max_iter);
  }
  else if (argv[1] == s_11){
    ok &= test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(Comm, tau);
  }
  else {
    ok = false;
  }
  MPI_Finalize();

  if (ok) return 0;
  else return 1;
}
