
#include "NNLS_solver.h"

namespace PHiLiP {

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    A_(A),
    Comm_(Comm), 
    b_(b), 
    x_(A.ColMap()), 
    LS_iter_(1000), 
    LS_tol_(10E-8), 
    Z(A_.NumMyCols()), 
    P(A_.NumMyCols()), 
    iter_solver_(false), 
    grad_exit_crit_(false) 
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumMyCols(), 0, A_.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
    }

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b,
  bool grad_exit_crit):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    A_(A),  
    Comm_(Comm), 
    b_(b), 
    x_(A.ColMap()), 
    LS_iter_(1000), 
    LS_tol_(10E-8), 
    Z(A_.NumMyCols()), 
    P(A_.NumMyCols()), 
    iter_solver_(false), 
    grad_exit_crit_(grad_exit_crit) 
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumMyCols(), 0, A_.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
    }

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    A_(A), 
    Comm_(Comm), 
    b_(b), 
    x_(A.ColMap()), 
    LS_iter_(LS_iter), 
    LS_tol_(LS_tol),
    Z(A_.NumMyCols()), 
    P(A_.NumMyCols()), 
    iter_solver_(iter_solver), 
    grad_exit_crit_(false) 
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumMyCols(), 0, A_.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
    }

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool grad_exit_crit, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol):
    all_parameters(parameters_input),
    parameter_handler(parameter_handler_input),
    A_(A), 
    Comm_(Comm), 
    b_(b), 
    x_(A.ColMap()), 
    LS_iter_(LS_iter), 
    LS_tol_(LS_tol),
    Z(A_.NumMyCols()), 
    P(A_.NumMyCols()), 
    iter_solver_(iter_solver), 
    grad_exit_crit_(grad_exit_crit) 
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumMyCols(), 0, A_.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
    }

void NNLS_solver::Epetra_PermutationMatrix(Epetra_CrsMatrix &P_mat){
  // Fill diagonal matrix with ones in the positive set
  // No longer in use
  double posOne = 1.0;
  for(int i = 0; i < P_mat.NumMyCols(); i++){
    int GlobalRow = P_mat.GRID(i);
    if (P[i] == 1) {
      P_mat.InsertGlobalValues(GlobalRow, 1, &posOne , &i);
    }
  }
}

void NNLS_solver::PositiveSetMatrix(Epetra_CrsMatrix &P_mat){
  // Create matrix P_mat which contains the positive set of columns in A

  // Create map between index_set and the columns to be added to P_mat
 // int colMap[colMap_size];
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
  }
}

void NNLS_solver::SubIntoX(Epetra_Vector &temp){
  // Substitute new values into the solution vector
  // int colMap[x_.GlobalLength()];
  std::vector<int> colMap(A_.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < x_.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.GlobalLength(); j++){
    if (P[j] == 1) {
      x_[j] = temp[colMap[j]];
    }
  }
}

void NNLS_solver::AddIntoX(Epetra_Vector &temp, double alpha){
  // Add vector temp time scalar alpha into the vector x
  // int colMap[x_.GlobalLength()];
  std::vector<int> colMap(A_.NumGlobalCols());
  int numCol = 0;
  for(int j = 0; j < x_.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.GlobalLength(); j++){
    if (P[j] == 1) {
      x_[j] += alpha*(temp[colMap[j]] -x_[j]);
    }
  }
}

void NNLS_solver::moveToActiveSet(int idx){
  // Move index at idx into the Active Set (Z set)
  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 

  std::swap(index_set(idx), index_set(numInactive_ - 1));
  numInactive_--;
}

void NNLS_solver::moveToInactiveSet(int idx){
  // Move index at idx into the Inactive Set (P set)
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;

  std::swap(index_set(idx), index_set(numInactive_));
  numInactive_++;
}

bool NNLS_solver::solve(){
  iter_ = 0;
  numInactive_ = 0;

  // Pre-mult by A^T
  Epetra_CrsMatrix AtA(Epetra_DataAccess::View, A_.ColMap(), A_.NumMyCols());
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
      return true;
    }

    AtA.Multiply(false, x_, AtAx);
    gradient = Atb;
    gradient.Update(-1.0, AtAx, 1.0); // gradient = A^T * (b-A*x)
    // std::cout << gradient << std::endl;

    grad_col = *gradient(0);
    for(int i = 0; i < gradient.GlobalLength() ; ++i){
      grad_eig[i] = grad_col[i];
    }
    
    // Find the maximum element of the gradient in the active set
    // Move that variable to the inactive set
    int numActive = A_.NumGlobalCols() - numInactive_;
    int argmaxGradient = -1;
    const double maxGradient = grad_eig(index_set.tail(numActive)).maxCoeff(&argmaxGradient);
    argmaxGradient += numInactive_;

    residual = b_;
    A_.Multiply(false, x_, Ax);
    residual.Update(-1.0, Ax, 1.0); // residual = b - A*x
    // std::cout << residual << std::endl;
    double normRes[1];
    residual.Norm2(normRes);

    double normb[1];
    b_.Norm2(normb);
    // Exit Condition on the residual based on the norm of b
    if ((normRes[0]) <= (all_parameters->hyper_reduction_param.NNLS_tol * normb[0])){
      return true;
    }

    // Old exit condition dependent on the maxGradient
    if (grad_exit_crit_){
      if (maxGradient < all_parameters->hyper_reduction_param.NNLS_tol){
        std::cout << "Exited due to Gradient Criteria" << std::endl;
        return true;
      }
    }
    
    moveToInactiveSet(argmaxGradient);

    // INNER LOOP
    while(true){
      // Check if max. number of iterations is reached
      if (iter_ >= all_parameters->hyper_reduction_param.NNLS_max_iter){
        return false;
      }

      // Create matrix P_mat with columns from set P
      Epetra_Map Map(A_.NumGlobalRows(),0,Comm_);
      Epetra_Map ColMap(numInactive_,0,Comm_);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive_);
      PositiveSetMatrix(P_mat);
      P_mat.FillComplete(ColMap, Map);

      // Create temporary solution vector temp which is only the length of numInactive
      Epetra_Vector temp(P_mat.ColMap());

      // Set up normal equations
      Epetra_CrsMatrix PtP(Epetra_DataAccess::View, P_mat.ColMap(), P_mat.NumMyCols());
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
      //std::cout << temp << std::endl;
      iter_++; // The solve is expensive, so that is what we count as an iteration
      
      // Check feasability...
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      int infeasibleIdx = -1;
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
      eigen_assert(feasible || 0 <= infeasibleIdx);

      // If solution is feasible, exit to outer loop
      if (feasible){
        SubIntoX(temp);
        // std::cout << "sub temp: " << x_ << std::endl;
        break;
      }

      // Infeasible solution -> interpolate to feasible one
      AddIntoX(temp, alpha);
      // std::cout << "added with alpha: " << x_ << std::endl;

      // Remove the infeasibleIdx column from the inactive set
      moveToActiveSet(infeasibleIdx);
    }
    
  }
}

} // PHiLiP namespace
