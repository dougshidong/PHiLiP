
#include "NNLS_solver.h"

namespace PHiLiP {

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b): NNLS_solver(
    parameters_input,
    parameter_handler_input,
    A,
    Comm,
    b,
    false,
    false,
    1000,
    10E-8) {}

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b,
  bool grad_exit_crit): NNLS_solver(
    parameters_input,
    parameter_handler_input,
    A,
    Comm,
    b,
    grad_exit_crit,
    false,
    1000,
    10E-8) {}

NNLS_solver::NNLS_solver(
  const Parameters::AllParameters *const parameters_input,
  const dealii::ParameterHandler &parameter_handler_input,
  const Epetra_CrsMatrix &A, 
  Epetra_MpiComm &Comm, 
  Epetra_Vector &b, 
  bool iter_solver, 
  int LS_iter, 
  double LS_tol): NNLS_solver(
    parameters_input,
    parameter_handler_input,
    A,
    Comm,
    b,
    false,
    iter_solver,
    LS_iter,
    LS_tol) {}

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
    Comm_(Comm), 
    A_(allocateMatrixToSingleCore(A)), 
    b_(allocateVectorToSingleCore(b)), 
    x_(A_.ColMap()), 
    multi_x_(A.DomainMap()),
    LS_iter_(LS_iter), 
    LS_tol_(LS_tol),
    Z(A_.NumGlobalCols()), 
    P(A_.NumGlobalCols()), 
    iter_solver_(iter_solver), 
    grad_exit_crit_(grad_exit_crit) 
    {
      index_set = Eigen::VectorXd::LinSpaced(A_.NumGlobalCols(), 0, A_.NumGlobalCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
      Z.flip(); // All columns begin in the Z set (Active)
    }

void NNLS_solver::Epetra_PermutationMatrix(Epetra_CrsMatrix &P_mat){
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

void NNLS_solver::PositiveSetMatrix(Epetra_CrsMatrix &P_mat){
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
  }
}

void NNLS_solver::SubIntoX(Epetra_Vector &temp){
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

void NNLS_solver::AddIntoX(Epetra_Vector &temp, double alpha){
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
  const int rank = this->Comm_.MyPID();
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
      multi_x_ = allocateVectorToMultipleCores(this->x_);
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
        multi_x_ = allocateVectorToMultipleCores(this->x_);
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
        multi_x_ = allocateVectorToMultipleCores(this->x_);
        return true;
      }
    }
    
    moveToInactiveSet(argmaxGradient);

    // INNER LOOP 
    bool no_feasible_soln = true;
    while(no_feasible_soln){
      // Check if max. number of iterations is reached
      if (iter_ >= all_parameters->hyper_reduction_param.NNLS_max_iter){
        multi_x_ = allocateVectorToMultipleCores(this->x_);
        return false;
      } 
      // Create matrix P_mat with columns from set P
      Epetra_Map Map(A_.NumGlobalRows(),(rank == 0) ?  A_.NumGlobalRows() : 0, 0, Comm_);
      Epetra_Map ColMap(numInactive_,(rank == 0) ?  numInactive_ : 0, 0, Comm_);
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
        SubIntoX(temp);
        no_feasible_soln = false;
      }
      else{
        // Infeasible solution -> interpolate to feasible one
        AddIntoX(temp, alpha);

        // Remove the infeasibleIdx column from the inactive set
        moveToActiveSet(infeasibleIdx);
      }
    }
  }
}

Epetra_CrsMatrix NNLS_solver::allocateMatrixToSingleCore(const Epetra_CrsMatrix &A)
{
  // Gather Matrix Information
  const int A_rows = A.NumGlobalRows();
  const int A_cols = A.NumGlobalCols();
  const int rank = Comm_.MyPID(); 
  // Create new maps for one core and gather old maps
  Epetra_Map single_core_row_A (A_rows, (rank == 0) ?  A_rows : 0, 0 , Comm_);
  Epetra_Map single_core_col_A (A_cols, (rank == 0) ?  A_cols : 0, 0 , Comm_);
  Epetra_Map old_row_map_A = A.RowMap();
  Epetra_Map old_domain_map_A = A.DomainMap();
  // Create Epetra_importer object
  Epetra_Import A_importer(single_core_row_A,old_row_map_A);
  // Create new A matrix
  Epetra_CrsMatrix A_temp (Epetra_DataAccess::Copy, single_core_row_A, A_cols);
  // Load the data from matrix A (Multi core) into A_temp (Single core)
  A_temp.Import(A, A_importer, Epetra_CombineMode::Insert);
  A_temp.FillComplete(single_core_col_A,single_core_row_A);
  return A_temp;
}

Epetra_Vector NNLS_solver::allocateVectorToSingleCore(const Epetra_Vector &b)
{
  // Gather Vector Information
  const int rank = Comm_.MyPID();
  const int b_size = b.GlobalLength();
  // Create new map for one core and gather old map
  Epetra_Map single_core_b (b_size, (rank == 0) ? b_size : 0, 0, Comm_);
  Epetra_BlockMap old_map_b = b.Map();
  // Create Epetra_importer object
  Epetra_Import b_importer(single_core_b, old_map_b);
  // Create new b vector
  Epetra_Vector b_temp (single_core_b); 
  // Load the data from vector b (Multi core) into b_temp (Single core)
  b_temp.Import(b, b_importer, Epetra_CombineMode::Insert);
  return b_temp;
}

Epetra_Vector NNLS_solver::allocateVectorToMultipleCores(const Epetra_Vector &c)
{
  // Create new multi core map and gather old single core map
  Epetra_BlockMap old_map_c = c.Map();
  Epetra_BlockMap multi_core_c = this->multi_x_.Map();
  // Create Epetra_importer object
  Epetra_Import c_importer (multi_core_c, old_map_c);
  // Create new c vector
  Epetra_Vector c_temp (multi_core_c);
  // Load the data from vector c (Single core) into c_temp (Multicore)
  c_temp.Import(c, c_importer, Epetra_CombineMode::Insert);
  return c_temp;

}
} // PHiLiP namespace
