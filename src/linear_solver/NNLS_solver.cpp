
#include "NNLS_solver.h"
#include "helper_functions.h"

namespace PHiLiP {

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    false,
    Comm,
    b,
    false,
    false,
    1000,
    10E-8) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A,
  const bool is_input_A_matrix_transposed, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    is_input_A_matrix_transposed,
    Comm,
    b,
    false,
    false,
    1000,
    10E-8) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b,
  bool grad_exit_crit): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    false,
    Comm,
    b,
    grad_exit_crit,
    false,
    1000,
    10E-8) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A,
  const bool is_input_A_matrix_transposed, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b,
  bool grad_exit_crit): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    is_input_A_matrix_transposed,
    Comm,
    b,
    grad_exit_crit,
    false,
    1000,
    10E-8) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    false,
    Comm,
    b,
    false,
    iter_solver,
    LS_iter,
    LS_tol) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A,
  const bool is_input_A_matrix_transposed, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol): NNLSSolver(
    parameters_input,
    parameter_handler_input,
    A,
    is_input_A_matrix_transposed,
    Comm,
    b,
    false,
    iter_solver,
    LS_iter,
    LS_tol) {}

NNLSSolver::NNLSSolver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A,
  const bool is_input_A_matrix_transposed, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool grad_exit_crit, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    Comm_(Comm), 
    A_((Comm.NumProc() == 1) ?  transpose_matrix_on_single_core(A, is_input_A_matrix_transposed) : allocate_matrix_to_single_core(A, is_input_A_matrix_transposed)),
    b_((Comm.NumProc() == 1) ?  b : allocate_vector_to_single_core(b)), 
    x_(A_.ColMap()), 
    multi_x_((is_input_A_matrix_transposed) ? A.RowMap(): A.ColMap()),
    LS_iter_(LS_iter), 
    LS_tol_(LS_tol),
    Z(A_.NumGlobalCols()), 
    P(A_.NumGlobalCols()), 
    iter_solver_(iter_solver), 
    grad_exit_crit_(grad_exit_crit),
    is_input_A_matrix_transposed_(is_input_A_matrix_transposed)
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumGlobalCols(), 0, A_.NumGlobalCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
      //EpetraExt::RowMatrixToMatlabFile("C_multicore_consol", A_);
    }

void NNLSSolver::epetra_permutation_matrix(Epetra_CrsMatrix &P_mat){
  // Fill diagonal matrix with ones in the positive set
  // No longer in use
  double posOne = 1.0;
  for(int i = 0; i < P_mat.NumMyCols(); i++){
    int globalRow = P_mat.GRID(i);
    if (P[i] == 1) {
      P_mat.InsertGlobalValues(globalRow, 1, &posOne , &i);
    }
  }
}

void NNLSSolver::positive_set_matrix(Epetra_CrsMatrix &P_mat){
  // Create matrix P_mat which contains the positive set of columns in A

  // Create map between index_set and the columns to be added to P_mat
  std::vector<int> colMap(A_.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < A_.NumGlobalCols(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }

  // Fill Epetra_CrsMatrix P_mat with columns of A in set P
  for(int i =0; i < A_.NumGlobalRows(); i++){
    double *row = new double[A_.NumGlobalCols()];
    int numE;
    const int globalRow = A_.GRID(i);
    A_.ExtractGlobalRowCopy(globalRow, A_.NumGlobalCols(), numE , row);
    for(int j = 0; j < A_.NumGlobalCols(); j++){
      if (P[j] == 1) {
        P_mat.InsertGlobalValues(i, 1, &row[j] , &colMap[j]);
        
      }
    }
    delete[] row;
  }
}

void NNLSSolver::sub_into_x(Epetra_Vector &temp){
  // Substitute new values into the solution vector
  std::vector<int> colMap(A_.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < x_.MyLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.MyLength(); j++){
    if (P[j] == 1) {
      x_[j] = temp[colMap[j]];
    }
  }
}

void NNLSSolver::add_into_x(Epetra_Vector &temp, double alpha){
  // Add vector temp time scalar alpha into the vector x
  std::vector<int> colMap(A_.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < x_.MyLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.MyLength(); j++){
    if (P[j] == 1) {
      x_[j] += alpha*(temp[colMap[j]] -x_[j]);
    }
  }
}

void NNLSSolver::move_to_active_set(int idx){
  // Move index at idx into the Active Set (Z set)
  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 

  std::swap(index_set(idx), index_set(numInactive_ - 1));
  numInactive_--;
}

void NNLSSolver::move_to_inactive_set(int idx){
  // Move index at idx into the Inactive Set (P set)
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;

  std::swap(index_set(idx), index_set(numInactive_));
  numInactive_++;
}

bool NNLSSolver::solve(){
  const int rank = this->Comm_.MyPID();
  iter_ = 0;
  numInactive_ = 0;
  // Pre-mult by A^T
  Epetra_CrsMatrix AtA(Epetra_DataAccess::Copy, A_.ColMap(), A_.NumMyCols());
  EpetraExt::MatrixMatrix::Multiply(A_, true, A_, false, AtA);

  Epetra_Vector Atb (A_.ColMap());
  A_.Multiply(true, b_, Atb);

  Epetra_Vector AtAx (A_.ColMap());
  Epetra_Vector Ax (A_.RowMap());
  Epetra_MultiVector gradient (A_.ColMap(), 1);
  Epetra_MultiVector residual (A_.RowMap(), 1);
  Eigen::VectorXd grad_eig (gradient.GlobalLength());
  Epetra_Vector grad_col (A_.ColMap());

  // OUTER LOOP
  while(true){
    // Early exit if all variables are inactive, which breaks 'maxCoeff' below. 
    if (A_.NumGlobalCols() == numInactive_){
      multi_x_ = allocate_vector_to_multiple_cores(this->x_, this->multi_x_);
      return true;
    }
    AtA.Multiply(false, x_, AtAx);
    gradient = Atb;
    gradient.Update(-1.0, AtAx, 1.0); // gradient = A^T * (b-A*x)

    grad_col = *gradient(0);
    for(int i = 0; i < gradient.MyLength() ; ++i){
      grad_eig[i] = grad_col[i];
    }

    // Find the maximum element of the gradient in the active set
    // Move that variable to the inactive set
    int numActive = A_.NumGlobalCols() - numInactive_;
    int argmaxGradient = -1;
    double maxGradient;
    if(rank==0){
      maxGradient = grad_eig(index_set.tail(numActive)).maxCoeff(&argmaxGradient);
    }
    MPI_Bcast(&maxGradient,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    argmaxGradient += numInactive_;
    MPI_Bcast(&argmaxGradient,1, MPI_INT,0,MPI_COMM_WORLD);
    residual = b_;
    A_.Multiply(false, x_, Ax);
    residual.Update(-1.0, Ax, 1.0); // residual = b - A*x
    double normRes[1];
    residual.Norm2(normRes);
    double normb[1];
    b_.Norm2(normb);
    // Old exit condition dependent on the maxGradient
    // Original exit condition presented in "SOLVING LEAST SQUARES PROBLEMS", by Charles L. Lawson and
    // Richard J. Hanson, Prentice-Hall, 1974
    // https://epubs.siam.org/doi/10.1137/1.9781611971217
    if (grad_exit_crit_){
      if (maxGradient < all_parameters->hyper_reduction_param.NNLS_tol){
        std::cout << "Exited due to Gradient Criteria" << std::endl;
        std::cout << "Norm-2 of b" << std::endl;
        std::cout << normb[0] << std::endl;
        std::cout << "Norm-2 of the residual (b-A*x)" << std::endl;
        std::cout << normRes[0] << std::endl;
        multi_x_ = allocate_vector_to_multiple_cores(this->x_, this->multi_x_);
        return true;
      }
    }
    // Exit Condition on the residual based on the norm of b
    // Updated exit condition presented in "Accelerated mesh sampling for the hyper reduction of nonlinear computational models",
    // by Chapman et. al, 2016 (EQUATION 13)
    // https://onlinelibrary.wiley.com/doi/full/10.1002/nme.5332
    else {
      if ((normRes[0]) <= (all_parameters->hyper_reduction_param.NNLS_tol * normb[0])){
        std::cout << "Exited due to Residual Criteria" << std::endl;
        std::cout << "Norm-2 of b" << std::endl;
        std::cout << normb[0] << std::endl;
        std::cout << "Norm-2 of the residual (b-A*x)" << std::endl;
        std::cout << normRes[0] << std::endl;
        multi_x_ = allocate_vector_to_multiple_cores(this->x_, this->multi_x_);
        return true;
      }
    }
    
    move_to_inactive_set(argmaxGradient);

    // INNER LOOP 
    bool no_feasible_soln = true;
    while(no_feasible_soln){
      // Check if max. number of iterations is reached
      if (iter_ >= all_parameters->hyper_reduction_param.NNLS_max_iter){
        multi_x_ = allocate_vector_to_multiple_cores(this->x_, this->multi_x_);
        return false;
      } 
      // Create matrix P_mat with columns from set P
      Epetra_Map Map(A_.NumGlobalRows(),(rank == 0) ?  A_.NumGlobalRows() : 0, 0, Comm_);
      Epetra_Map ColMap(numInactive_,(rank == 0) ?  numInactive_ : 0, 0, Comm_);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive_);
      positive_set_matrix(P_mat);
      P_mat.FillComplete(ColMap, Map);
      // Create temporary solution vector temp which is only the length of numInactive
      Epetra_Vector temp(P_mat.ColMap());

      // Set up normal equations
      Epetra_CrsMatrix PtP(Epetra_DataAccess::Copy, P_mat.ColMap(), P_mat.NumMyCols());
      EpetraExt::MatrixMatrix::Multiply(P_mat, true, P_mat, false, PtP);

      Epetra_Vector Ptb (P_mat.ColMap());
      P_mat.Multiply(true, b_, Ptb);
      // An iterative solver can be used by setting iter_solver_ to true
      if (iter_solver_){
        // Solve least-squares problem in inactive set only
        Epetra_LinearProblem problem(&P_mat, &temp, &b_);
        AztecOO solver(problem);

        // Iterative Solver Setup
        solver.SetAztecOption(AZ_conv, AZ_rhs);
        solver.SetAztecOption( AZ_precond, AZ_Jacobi);
        solver.SetAztecOption(AZ_output, AZ_none);
        solver.Iterate(LS_iter_, LS_tol_);
      }
      else{
        // Solve least-squares problem in inactive set only
        Epetra_LinearProblem problem(&PtP, &temp, &Ptb);

        // Direct Solver Setup
        Amesos Factory;
        std::string SolverType = "Klu";
        std::unique_ptr<Amesos_BaseSolver> Solver(Factory.Create(SolverType, problem));

        Teuchos::ParameterList List;
        Solver->SetParameters(List);
        Solver->SymbolicFactorization();
        Solver->NumericFactorization();
        Solver->Solve();
      }
      iter_++; // The solve is expensive, so that is what we count as an iteration
      
      // Check feasability...
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      int infeasibleIdx = -1;
      if(rank == 0){ // Will only proceed on the root as the other cores do no have access to this information
        for(int k = 0; k < numInactive_; k++){
          int idx = index_set[k];
          if (temp[k] < 0){
            // t should always be in [0,1]
            double t = -x_[idx]/(temp[k] - x_[idx]);
            if (alpha > t){
              alpha = t;
              infeasibleIdx = k;
              feasible = false;
            }
          }
        }
      }
      // Broadcast these variables so that the loop does not fail on other cores
      MPI_Bcast(&feasible,1,MPI_CXX_BOOL,0,MPI_COMM_WORLD);
      MPI_Bcast(&infeasibleIdx, 1, MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&alpha,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      eigen_assert(feasible || 0 <= infeasibleIdx);

      // If solution is feasible, exit to outer loop
      if (feasible){
        sub_into_x(temp);
        no_feasible_soln = false;
      }
      else{
        // Infeasible solution -> interpolate to feasible one
        add_into_x(temp, alpha);

        // Remove the infeasibleIdx column from the inactive set
        move_to_active_set(infeasibleIdx);
      }
    }
  }
}

} // PHiLiP namespace
