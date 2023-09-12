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
#include <eigen/test/random_matrix_helper.h>

using namespace Eigen;
/// @brief Fills the entries in an empty Eigen::MatrixXd from an Epetra_Vector structure
/// @param col length of the vector 
/// @param x Full Epetra vector to copy
/// @param x_eig Empty Eigen::MatrixXd
void epetra_to_eig_vec(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig){
  // Convert epetra vector to eigen vector
  for(int i = 0; i < col; i++){
    x_eig(i,0) = x[i];
  }
}

/// @brief Returns an Epetra_CrsMatrix with the entries from an Eigen::MatrixXd structure
/// @param A_eig Full Eigen Matrix to copy
/// @param col number of columns
/// @param row number of rows
/// @param Comm MpiComm for Epetra Maps
/// @return Full Epetra_CrsMatrixss
Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, int col, int row, Epetra_MpiComm &Comm){
  // Create an empty Epetra structure with the right dimensions
  Epetra_Map RowMap(row,0,Comm);
  Epetra_Map ColMap(col,0,Comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, RowMap, col);
  const int numMyElements = RowMap.NumGlobalElements();

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
