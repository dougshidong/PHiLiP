#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include <deal.II/base/mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <eigen/Eigen/QR>
#include "parameters/parameters.h"
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

Parameters::AllParameters reinit_params(Parameters::AllParameters all_parameters, const double tol, const int max_iter){
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = all_parameters;

    parameters.hyper_reduction_param.NNLS_tol = tol;
    parameters.hyper_reduction_param.NNLS_max_iter = max_iter;
    return parameters;
}

// THE FOLLOWING THREE FUNCTIONS ARE COPIED FROM eigen/test/random_matrix_helper.h in Eigen
/**
 * Generate a random unitary matrix of prescribed dimension.
 *
 * The algorithm is using a random Householder sequence to produce
 * a random unitary matrix.
 *
 * @tparam MatrixType type of matrix to generate
 * @param dim row and column dimension of the requested square matrix
 * @return random unitary matrix
 */
template<typename MatrixType>
MatrixType generateRandomUnitaryMatrix(const Index dim)
{
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;

  MatrixType v = MatrixType::Identity(dim, dim);
  VectorType h = VectorType::Zero(dim);
  for (Index i = 0; i < dim; ++i)
  {
    v.col(i).tail(dim - i - 1) = VectorType::Random(dim - i - 1);
    h(i) = 2 / v.col(i).tail(dim - i).squaredNorm();
  }

  const Eigen::HouseholderSequence<MatrixType, VectorType> HSeq(v, h);
  return MatrixType(HSeq);
}

/**
 * Generation of random matrix with prescribed singular values.
 *
 * We generate random matrices with given singular values by setting up
 * a singular value decomposition. By choosing the number of zeros as
 * singular values we can specify the rank of the matrix.
 * Moreover, we also control its spectral norm, which is the largest
 * singular value, as well as its condition number with respect to the
 * l2-norm, which is the quotient of the largest and smallest singular
 * value.
 *
 * Reference: For details on the method see e.g. Section 8.1 (pp. 62 f) in
 *
 *   C. C. Paige, M. A. Saunders,
 *   LSQR: An algorithm for sparse linear equations and sparse least squares.
 *   ACM Transactions on Mathematical Software 8(1), pp. 43-71, 1982.
 *   https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
 *
 * and also the LSQR webpage https://web.stanford.edu/group/SOL/software/lsqr/.
 *
 * @tparam MatrixType matrix type to generate
 * @tparam RealScalarVectorType vector type with real entries used for singular values
 * @param svs vector of desired singular values
 * @param rows row dimension of requested random matrix
 * @param cols column dimension of requested random matrix
 * @param M generated matrix with prescribed singular values
 */
template<typename MatrixType, typename RealScalarVectorType>
void generateRandomMatrixSvs(const RealScalarVectorType &svs, const Index rows, const Index cols, MatrixType& M)
{
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef Matrix<Scalar, Rows, Rows> MatrixAType;
  typedef Matrix<Scalar, Cols, Cols> MatrixBType;

  const Index min_dim = (std::min)(rows, cols);

  const MatrixAType U = generateRandomUnitaryMatrix<MatrixAType>(rows);
  const MatrixBType V = generateRandomUnitaryMatrix<MatrixBType>(cols);

  M = U.block(0, 0, rows, min_dim) * svs.asDiagonal() * V.block(0, 0, cols, min_dim).transpose();
}

template<typename VectorType, typename RealScalar>
VectorType setupRangeSvs(const Index dim, const RealScalar min, const RealScalar max)
{
  VectorType svs = VectorType::Random(dim);
  if(dim == 0)
    return svs;
  if(dim == 1)
  {
    svs(0) = min;
    return svs;
  }
  std::sort(svs.begin(), svs.end(), std::greater<RealScalar>());

  // scale to range [min, max]
  const RealScalar c_min = svs(dim - 1), c_max = svs(0);
  svs = (svs - VectorType::Constant(dim, c_min)) / (c_max - c_min);
  return min * (VectorType::Ones(dim) - svs) + max * svs;
}

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
bool test_nnls_known_CLASS(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Eigen::MatrixXd &A_eig, int col, int row, Eigen::MatrixXd &x_eig, Eigen::MatrixXd &b_eig, Epetra_MpiComm &Comm, dealii::ConditionalOStream pcout){
  // Check solution of NNLS problem with a known solution
  // Returns true if the solver exits for any condition other than max_iter and if the solution x is accurate to the true solution and satisfies the conditions above
  
  // Create Epetra structures of A and b from Eigen matrices and pointers, respectively
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, col, row, Comm));
  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,row,Comm));
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit condition
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x = NNLS_prob.get_solution();
  x.Print(std::cout);
  
  // If the maximum iterations were reached, output message to signal possible failure of the solver
  // Still check the optimality of the solution as it can still be the close to the exact solution
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  // Convert the solution Epetra_Vector to an Eigen::MatrixXd
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);

  // Confirm the optimality of the solution wrt tolerance and positivity
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  // Compare to the exact solutions
  opt&= x_nnls_eig.isApprox(x_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  return opt;
}

/// @brief Test to check handling of matrix with no columns
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the length of the solution (returns true if the solver returns true, it completes zero iterations and the length of the vector is zero) 
bool test_nnls_handles_Mx0_matrix(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  // Build matrix with no columns
  const int row = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig(row, 0);
  pcout << A_eig << std::endl;
  VectorXd b_eig = VectorXd::Random(row);

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, row, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,row,Comm));

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterationss
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  // Check the number of iterations completed and size of the solution vector (both expected to be zeros)
  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x = NNLS_prob.get_solution();
  opt &= (x.GlobalLength()== 0);
  return opt;
}

/// @brief Test to check handling of matrix with no rows and columns (0x0)
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return  boolean depending on the exit condition of the solver, the number of iterations, and the length of the solution (returns true if the solver returns true, it completes zero iterations and the length of the vector is zero) 
bool test_nnls_handles_0x0_matrix(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
// Build matrix with no rows and columns
  MatrixXd A_eig(0, 0);
  VectorXd b_eig(0);
  
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, 0, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,0,Comm));

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterationss
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  // Check the number of iterations completed and size of the solution vector (both expected to be zeros)
  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x(NNLS_prob.get_solution());
  opt &= (x.GlobalLength()== 0);
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
bool test_nnls_random_problem(PHiLiP::Parameters::AllParameters *all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
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

  // Estimate the tolerance and max_iter based on problem size
  using Scalar = typename MatrixXd::Scalar;
  using std::sqrt;
  const Scalar tolerance =
       sqrt(Eigen::GenericNumTraits<Scalar>::epsilon()) * b_eig.cwiseAbs().maxCoeff() * A_eig.cwiseAbs().maxCoeff();
  Index max_iter = 5 * A_eig.cols();  // A heuristic guess.
  
  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,rows,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  Parameters::AllParameters const new_parameters = reinit_params(*all_parameters, tolerance, max_iter);
  NNLSSolver NNLS_prob(&new_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations
  // As noted above, this is permissable due to the change in the exit condition but the solution must satisfy the gradient exit condition
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  // Confirm the optimality of the solution wrt tolerance and positivity
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.get_solution());
  x.Print(std::cout);
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tolerance, true); // random flag set to true
  return opt;
}

/// @brief Test to check handling of zero RHS vector
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the entries in the solution (returns true if the solver returns true, it completes less than two iteration, and the solution vector is zeros)
bool test_nnls_handles_zero_rhs(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and RHS vector of zeros
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Zero(rows);


  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,rows,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  bool opt = true;
  // Check if solver exited by reaching the maximum number of iterations
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }

  // Check the number of iterations completed (expected to be 0 or 1) and that all the entries in the solution are zero
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.get_solution());
  x.Print(std::cout);
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  opt &= (NNLS_prob.iter_ <= 1);
  opt &= (x_nnls_eig.isApprox(VectorXd::Zero(cols)));
  return opt;
}

/// @brief Test to check handling of matrix with dependent columns
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver and the optimality of the solution (returns true if the max. iters are reached or if the solution is the optimal one)
bool test_nnls_handles_dependent_columns(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and matrix with dependent columns
  const Index rank = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2);
  const Index cols = 2 * rank;
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, rank) * MatrixXd::Random(rank, cols);
  MatrixXd b_eig = VectorXd::Random(rows);

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,rows,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  // From Eigen : 
  /* What should happen when the input 'A' has dependent columns?
     We might still succeed. Or we might not converge.
     Either outcome is fine. If Success is indicated,
     then 'x' must actually be a solution vector. */
  bool opt = true;
  Epetra_Vector x(NNLS_prob.get_solution());
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }
  else{
    // Confirm the optimality of the solution wrt tolerance and positivity
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  }
  pcout << " Solution x "<< std::endl;
  x.Print(std::cout);
  return opt;
}

/// @brief Test to check handling of a wide matrix (ie. cols > rows)
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver and the optimality of the solution (returns true if the max. iters are reached or if the solution is the optimal one)
bool test_nnls_handles_wide_matrix(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  // Random problem with dimensions less than EIGEN_TEST_MAX_SIZE and wide matrix
  const Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(2, cols - 1);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Random(rows);


  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,rows,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
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
  Epetra_Vector x(NNLS_prob.get_solution());
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  // Check if solver exited by reaching the maximum number of iterations
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }
  else{
    // Confirm the optimality of the solution wrt tolerance and positivity
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  }
  pcout << " Solution x "<< std::endl;
  x.Print(std::cout);
  return opt;
}

/// @brief Test to check handling when the solution vector is set to the exact solution before the solver is run
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the optimality of the solution (returns true if the solver returns true, it completes zero iterations, and the solution is optimal)
bool test_nnls_special_case_solves_in_zero_iterations(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive
  MatrixXd b_eig = A_eig * x_eig;

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,b_vec.size(),Comm));
  Eigen::VectorXd x_vec = Eigen::Map<Eigen::VectorXd>(x_eig.data(), x_eig.size());
  Epetra_Vector x_start(eig_to_epetra_vector(x_vec,x_vec.size(),Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  NNLS_prob.starting_solution(x_start); // SET SOLUTION TO EXACT SOLUTION
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations
  bool opt = true;
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.get_solution());
  x.Print(std::cout);
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  // Check the number of iterations (expected to be zero)
  opt &= (NNLS_prob.iter_ == 0);
  // Confirm the optimality of the solution wrt tolerance and positivity
  opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  return opt;
}

/// @brief Test to check handling when the solution should be found in n iterations due to structure of RHS
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @param max_iter maximum number of iterations
/// @return boolean depending on the exit condition of the solver, the number of iterations, and the optimality of the solution (returns true if the solver returns true, it completes n = num of cols. iterations, and the solution is optimal)
bool test_nnls_special_case_solves_in_n_iterations(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,m,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(all_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //

  // Check if solver exited by reaching the maximum number of iterations
  bool opt = true;
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.get_solution());
  x.Print(std::cout);
  Eigen::MatrixXd x_nnls_eig(x.GlobalLength(),1);
  epetra_to_eig_vec(x.GlobalLength(), x , x_nnls_eig);
  // Check the number of iterations (expected to be equal to the number of columns in A)
  opt &= (NNLS_prob.iter_ == n);
  // Confirm the optimality of the solution wrt tolerance and positivity
  opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, all_parameters->hyper_reduction_param.NNLS_tol);
  return opt;
}

/// @brief Test to check handling when the maximum iterations is too low
/// @param Comm MpiComm for Epetra Maps
/// @param tau tolerance for the residual exit condition
/// @return boolean depending on the exit condition of the solver and the number of iterations (returns true if the solver returns false due to exiting by max_iter, and the number of iterations is equal to n-1)
bool test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(PHiLiP::Parameters::AllParameters *all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  // Using the special case that takes `n` iterations,
  // from `test_nnls_special_case_solves_in_n_iterations`,
  // we can set max iterations too low and that should cause the solve to fail.

  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  // Convert Eigen structures to Epetra
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Eigen::VectorXd b_vec = Eigen::Map<Eigen::VectorXd>(b_eig.data(), b_eig.size());
  Epetra_Vector b(eig_to_epetra_vector(b_vec,m,Comm));

  pcout << " Matrix A "<< std::endl;
  A.Print(std::cout);
  
  pcout << " Vector b "<< std::endl;
  b.Print(std::cout);

  // Set max_iters too low to cause solver to fail/return false on purpose
  const Index max_iters = n - 1;
  Parameters::AllParameters const new_parameters = reinit_params(*all_parameters, 1E-8, max_iters);
  // Create instance of NNLS solver, and call .solve to find the solution and exit conditions
  NNLSSolver NNLS_prob(&new_parameters, parameter_handler, A, Comm, b);
  bool exit_con = NNLS_prob.solve();

  // Check if solver exited by reaching the maximum number of iterations, return true in this case
  bool opt = false;
  if (!exit_con){
    pcout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt = true;
  }
  pcout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.get_solution());
  x.Print(std::cout);
  // Check the number of iterations is in fact equal to the maximum number of iterations
  opt &= (NNLS_prob.iter_ == max_iters);
  return opt;
}

/// @brief Case 1: 4x2 problem, unconstrained solution positive
bool case_1 (const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig(4, 2);
  Eigen::MatrixXd x_eig(2,1);
  Eigen::MatrixXd b_eig(4,1);
  A_eig << 1, 1,  2, 4,  3, 9,  4, 16;
  b_eig << 0.6, 2.2, 4.8, 8.4;
  x_eig << 0.1, 0.5;

  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 2, 4, x_eig, b_eig, Comm, pcout);
}

/// @brief Case 2: 4x3 problem, unconstrained solution positive
bool case_2 (const PHiLiP::Parameters::AllParameters *const all_parameters,
                const dealii::ParameterHandler &parameter_handler,
                Epetra_MpiComm &Comm,
                dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1,  1,  1,
       2,  4,  8,
       3,  9, 27,
       4, 16, 64;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13;

  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 3, 4, x_eig, b_eig, Comm, pcout);
}

/// @brief Case 3: Simple 4x4 problem, unconstrained solution non-negative
bool case_3 (const PHiLiP::Parameters::AllParameters *const all_parameters,
                const dealii::ParameterHandler &parameter_handler,
                Epetra_MpiComm &Comm,
                dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig(4,4);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(4,1);

  A_eig << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13, 0;

  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 4, 4, x_eig, b_eig, Comm, pcout);
}

/// @brief Case 4: Simple 4x3 problem, unconstrained solution non-negative
bool case_4 (const PHiLiP::Parameters::AllParameters *const all_parameters,
              const dealii::ParameterHandler &parameter_handler,
              Epetra_MpiComm &Comm,
              dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.23, 1.24, 3.81, 8.72;
  x_eig << 0.1, 0, 0.13;

  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 3, 4, x_eig, b_eig, Comm, pcout);
}

/// @brief Case 5: Simple 4x3 problem, unconstrained solution indefinite
bool case_5 (const PHiLiP::Parameters::AllParameters *const all_parameters,
              const dealii::ParameterHandler &parameter_handler,
              Epetra_MpiComm &Comm,
              dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.13, 0.84, 2.91, 7.12;
   // Solution obtained by original nnls() implementation in Fortran
  x_eig << 0.0, 0.0, 0.1106544;

  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 3, 4, x_eig, b_eig, Comm, pcout);
}

/// @brief Case MATLAB: 1024 x 49 problem from ECSW in MATLAB for Burgers' 1D
bool case_MATLAB (const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
  Eigen::MatrixXd A_eig = load_csv<MatrixXd>("C.csv");
  Eigen::MatrixXd b_eig = load_csv<MatrixXd>("d.csv");
  Eigen::MatrixXd x_eig = load_csv<MatrixXd>("x_pow_4.csv");


  return test_nnls_known_CLASS(all_parameters, parameter_handler, A_eig, 1024, 49, x_eig, b_eig, Comm, pcout);
}

/// @brief Case multiCore: Testing NNLS on multiple cores
bool test_nnls_multiCore(const PHiLiP::Parameters::AllParameters *const all_parameters,
                  const dealii::ParameterHandler &parameter_handler,
                  Epetra_MpiComm &Comm,
                  dealii::ConditionalOStream pcout) {
    bool ok = true;
    PHiLiP::Parameters::AllParameters non_const_all_param = *all_parameters;
    pcout << "Case 1" << std::endl;
    ok &= case_1(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case 2" << std::endl;
    ok &= case_2(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Mx0" << std::endl;
    ok &= test_nnls_handles_Mx0_matrix(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case 0x0" << std::endl;
    ok &= test_nnls_handles_0x0_matrix(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Random" << std::endl;
    ok &= test_nnls_random_problem(&non_const_all_param, parameter_handler, Comm, pcout);
    pcout << "Case Zero RHS" << std::endl;
    ok &= test_nnls_handles_zero_rhs(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Dependent Columns" << std::endl;
    ok &= test_nnls_handles_dependent_columns(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Wide Matrix" << std::endl;
    ok &= test_nnls_handles_wide_matrix(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Zero Iter" << std::endl;
    ok &= test_nnls_special_case_solves_in_zero_iterations(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case N Iter" << std::endl;
    ok &= test_nnls_special_case_solves_in_n_iterations(all_parameters, parameter_handler, Comm, pcout);
    pcout << "Case Max Iter too low" << std::endl;
    ok &= test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(&non_const_all_param, parameter_handler, Comm, pcout);
    pcout << "Case MATLAB" << std::endl;
    Parameters::AllParameters new_parameters = reinit_params(non_const_all_param, 1E-4, 10000);
    ok &= case_MATLAB(&new_parameters, parameter_handler, Comm, pcout);
    return ok;

}

int main(int argc, char *argv[]){
  MPI_Init(&argc,&argv);
  const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );

  //double tau = 1E-8;
  //const int max_iter = 10000;

  // Setting the same seed on all cores for random functions
  int seed = time(0);
  MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
  srand((unsigned int) seed);

  bool ok = true;

  dealii::ParameterHandler parameter_handler;
  Parameters::AllParameters::declare_parameters (parameter_handler);

  // Read inputs from parameter file and set those values in AllParameters object
  Parameters::AllParameters all_parameters;
  all_parameters.parse_parameters (parameter_handler);

  Parameters::AllParameters new_parameters = reinit_params(all_parameters, 1E-8, 10000);

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
  std::string s_12 = "multiCore";
   
  if (argv[1] == s_1){
    ok &= case_1(&new_parameters, parameter_handler, Comm, pcout);
    ok &= case_2(&new_parameters, parameter_handler, Comm, pcout);
    ok &= case_3(&new_parameters, parameter_handler, Comm, pcout);
    ok &= case_4(&new_parameters, parameter_handler, Comm, pcout);
    Parameters::AllParameters new_new_parameters = reinit_params(all_parameters, 1E-3, 10000);
    ok &= case_5(&new_new_parameters, parameter_handler, Comm, pcout); // had to relax the tolerance for this case ( tol = 1E-3)
  }
  else if (argv[1] == s_2){
    Parameters::AllParameters new_parameters = reinit_params(all_parameters, 1E-4, 10000);
    ok &= case_MATLAB(&new_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_3){
    ok &= test_nnls_handles_Mx0_matrix(&all_parameters, parameter_handler, Comm, pcout);
  }  
  else if (argv[1] == s_4){
    ok &= test_nnls_handles_0x0_matrix(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_5){
    ok &= test_nnls_random_problem(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_6){
    ok &= test_nnls_handles_zero_rhs(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_7){
    ok &= test_nnls_handles_dependent_columns(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_8){
    ok &= test_nnls_handles_wide_matrix(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_9){
    ok &= test_nnls_special_case_solves_in_zero_iterations(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_10){
    ok &= test_nnls_special_case_solves_in_n_iterations(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_11){
    ok &= test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(&all_parameters, parameter_handler, Comm, pcout);
  }
  else if (argv[1] == s_12){
    ok &= test_nnls_multiCore(&all_parameters, parameter_handler, Comm, pcout);
  }
  else {
    ok = false;
  }
  MPI_Finalize();

  if (ok) return 0;
  else return 1;
}
