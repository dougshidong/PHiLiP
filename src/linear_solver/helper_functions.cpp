#include "helper_functions.h"
#include <iostream>
#include <fstream>
#include <eigen/Eigen/QR>
#include <iostream>
#include <vector>
#include <Epetra_MpiComm.h>
#include <Epetra_ConfigDefs.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_LinearProblem.h>
#include <EpetraExt_MatrixMatrix.h>

using namespace Eigen;
/**  @brief Fills the entries in an empty Eigen::MatrixXd from an Epetra_Vector structure
*    @param col length of the vector 
*    @param x Full Epetra vector to copy
*    @param x_eig Empty Eigen::MatrixXd
*/
void epetra_to_eig_vec(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig){
  // Gather local information
  int local_size = x.MyLength();
  const Epetra_Comm& comm = x.Comm();
  int np = comm.NumProc();
  
  std::vector<double> x_values(col);
  std::vector<int> local_sizes(np);
  std::vector<int> displacements(np, 0);
  // Gather all local_size into local_sizes. Vector will be the same globally
  MPI_Allgather(&local_size, 1, MPI_INT,
                local_sizes.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  // Calculate the global coordinates necessary
  for (int i = 1; i < np; ++i){
    displacements[i] = displacements[i-1] + local_sizes[i-1];
  }
  // Store the Epetra_vector values into x_values
  MPI_Allgatherv(x.Values(), local_size, MPI_DOUBLE,
                x_values.data(), local_sizes.data(), displacements.data(), MPI_DOUBLE,
                MPI_COMM_WORLD);
  // Convert epetra vector to eigen vector
  for(int i = 0; i < col; i++){
    x_eig(i,0) = x_values[i];
  }
}

/**  @brief Returns an Epetra_CrsMatrix with the entries from an Eigen::MatrixXd structure
*    @param A_eig Full Eigen Matrix to copy
*    @param col number of columns
*    @param row number of rows
*    @param Comm MpiComm for Epetra Maps
*    @return Full Epetra_CrsMatrix
*/
Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, int col, int row, Epetra_MpiComm &Comm){
  // Create an empty Epetra structure with the right dimensions
  Epetra_Map RowMap(row,0,Comm);
  Epetra_Map ColMap(col,0,Comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, RowMap, col);
  const int numMyElements = RowMap.NumMyElements();

  // Fill the Epetra_CrsMatrix from the Eigen::MatrixXd
  for (int localRow = 0; localRow < numMyElements; ++localRow){
      const int globalRow = RowMap.GID(localRow);
      for(int n = 0 ; n < A_eig.cols() ; n++){
          A.InsertGlobalValues(globalRow, 1, &A_eig(globalRow, n), &n);
      }
  }

  A.FillComplete(ColMap, RowMap);
  return A;
}

MatrixXd epetra_to_eig_matrix(Epetra_CrsMatrix A_epetra){
  // Create an empty Eigen structure
  MatrixXd A(A_epetra.NumGlobalRows(), A_epetra.NumGlobalCols());
  // Fill the Eigen::MatrixXd from the Epetra_CrsMatrix
  for (int m = 0; m < A_epetra.NumGlobalRows(); m++) {
      double *row = A_epetra[m];
      for (int n = 0; n < A_epetra.NumGlobalCols(); n++) {
          A(m,n) = row[n];
      }
  }
  return A;
}

/** @brief Returns an Epetra_Vector with entries from an Eigen::Vector structure
*   @param a_eigen Eigen Vector to copy
*   @param size size of vector
*   @param MpiComm for Epetra Maps
*   @return Epetra_Vector
*/
Epetra_Vector eig_to_epetra_vector(Eigen::VectorXd &a_eigen, int size, Epetra_MpiComm &Comm){
  // Create an Epetra Vector distributed along all cores in Comm
  Epetra_Map vecMap(size,0,Comm);
  Epetra_Vector a_epetra(vecMap);
  // Fill the Epetra_Vector with values from the Eigen Vector
  const int numMyElements = vecMap.NumMyElements();
  for (int localElement = 0; localElement < numMyElements; localElement++){
    const int globalElement = vecMap.GID(localElement);
    a_epetra.ReplaceGlobalValues(1, &a_eigen(globalElement), &globalElement);
  }
  return a_epetra;
}