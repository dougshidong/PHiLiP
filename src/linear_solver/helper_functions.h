#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

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

/// Helper functions for transferring information between Epetra and Eigen structures

using namespace Eigen;
/// @brief Fills the entries in an empty Eigen::MatrixXd from an Epetra_Vector structure
/// @param col length of the vector 
/// @param x Full Epetra vector to copy
/// @param x_eig Empty Eigen::MatrixXd
void epetra_to_eig_vec(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig);

/// @brief Returns an Epetra_CrsMatrix with the entries from an Eigen::MatrixXd structure
/// @param A_eig Full Eigen Matrix to copy
/// @param col number of columns
/// @param row number of rows
/// @param Comm MpiComm for Epetra Maps
/// @return Full Epetra_CrsMatrixss
Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, int col, int row, Epetra_MpiComm &Comm);

/// @brief Load data from CSV file into an Eigen Matrix of type M
/// @tparam M type of Eigen matrix
/// @param path filepath to csv
/// @return filled matrix of type M
template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    if (! indata.is_open()){
      indata.open("../tests/unit_tests/linear_solver/"+ path);
    }

    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

MatrixXd epetra_to_eig_matrix(Epetra_CrsMatrix A_epetra);

#endif