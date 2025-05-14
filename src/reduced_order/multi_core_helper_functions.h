#ifndef __MULTI_CORE_HELPER_FUNCTIONS_H__
#define __MULTI_CORE_HELPER_FUNCTIONS_H__

#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <iostream>
#include <memory>

/** @brief Allocates the incoming b vector to a single core
*   @param b Epetra_Vector to reallocate on a single core
*   @return An Epetra_Vector with the data of b distributed on a single core
*/ 
Epetra_Vector allocate_vector_to_single_core(const Epetra_Vector &b);

/** @brief Re-allocates solution to multiple cores
*   @param c Epetra_vector to reallocate on multiple cores
*   @return An Epetra_vector with the data of c distributed on multiple cores
*/
Epetra_Vector allocate_vector_to_multiple_cores(const Epetra_Vector &c, Epetra_Vector &multi_x_);

/** @brief Allocates the incoming A Matrix to a single core
*   @param A Epetra_CrsMatrix to reallocate on a single core
*   @return An Epetra_CrsMatrix with the data of A distributed on a single core
*/
Epetra_CrsMatrix allocate_matrix_to_single_core(const Epetra_CrsMatrix &A, const bool is_input_A_matrix_transposed);

/// @brief Tranpose matrix on one core
/// @param A 
/// @param is_input_A_matrix_transposed 
/// @return  An Epetra_CrsMatrix with the data of A transposed on a single core
Epetra_CrsMatrix transpose_matrix_on_single_core(const Epetra_CrsMatrix &A, const bool is_input_A_matrix_transposed);

/// Copy all elements in vector b to all cores
Epetra_Vector copy_vector_to_all_cores(const Epetra_Vector &b);

/// Copy all elements in matrix A to all cores
Epetra_CrsMatrix copy_matrix_to_all_cores(const Epetra_CrsMatrix &A);

#endif