#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_Import.h>
#include "multi_core_helper_functions.h"

Epetra_Vector allocate_vector_to_single_core(const Epetra_Vector &b){
    // Gather Vector Information
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    const int rank = Comm.MyPID();
    const int b_size = b.GlobalLength();
    // Create new map for one core and gather old map
    Epetra_Map single_core_b (b_size, (rank == 0) ? b_size : 0, 0, Comm);
    Epetra_BlockMap old_map_b = b.Map();
    // Create Epetra_importer object
    Epetra_Import b_importer(single_core_b, old_map_b);
    // Create new b vector
    Epetra_Vector b_temp (single_core_b); 
    // Load the data from vector b (Multi core) into b_temp (Single core)
    b_temp.Import(b, b_importer, Epetra_CombineMode::Insert);
    return b_temp;
}

Epetra_Vector allocate_vector_to_multiple_cores(const Epetra_Vector &c, Epetra_Vector &multi_x_)
{
  // Create new multi core map and gather old single core map
  Epetra_BlockMap old_map_c = c.Map();
  Epetra_BlockMap multi_core_c = multi_x_.Map();
  // Create Epetra_importer object
  Epetra_Import c_importer (multi_core_c, old_map_c);
  // Create new c vector
  Epetra_Vector c_temp (multi_core_c);
  // Load the data from vector c (Single core) into c_temp (Multicore)
  c_temp.Import(c, c_importer, Epetra_CombineMode::Insert);
  return c_temp;

}

Epetra_CrsMatrix allocate_matrix_to_single_core(const Epetra_CrsMatrix &A, const bool is_input_A_matrix_transposed = false){
    // Gather Matrix Information
    const int A_rows = A.NumGlobalRows();
    const int A_cols = A.NumGlobalCols();
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    const int rank = Comm.MyPID(); 
    // Create new maps for one core and gather old maps
    Epetra_Map single_core_row_A (A_rows, (rank == 0) ?  A_rows : 0, 0 , Comm);
    Epetra_Map single_core_col_A (A_cols, (rank == 0) ?  A_cols : 0, 0 , Comm);
    Epetra_Map old_row_map_A = A.RowMap();
    Epetra_Map old_domain_map_A = A.DomainMap();
    // Create Epetra_importer object
    Epetra_Import A_importer(single_core_row_A,old_row_map_A);
    // Create new A matrix
    Epetra_CrsMatrix A_temp (Epetra_DataAccess::Copy, single_core_row_A, A_cols);
    // Load the data from matrix A (Multi core) into A_temp (Single core)
    A_temp.Import(A, A_importer, Epetra_CombineMode::Insert);
    A_temp.FillComplete(single_core_col_A,single_core_row_A);
  
    std::shared_ptr<Epetra_CrsMatrix> A_ptr;
    if (is_input_A_matrix_transposed) {
      // Tranpose matrix onto one core
      Epetra_CrsMatrix A_trans(Epetra_DataAccess::Copy, single_core_col_A, A_rows);
      if (rank == 0){
        for (int i = 0; i < A_rows; i++){
          int maxEntries = A_temp.MaxNumEntries();
          std::vector<double> values(maxEntries);
          std::vector<int> indices(maxEntries);
          int numEntries;
          
          A_temp.ExtractGlobalRowCopy(i, maxEntries, numEntries, values.data(), indices.data());
          for (int j = 0; j < numEntries; ++j) {
            int row = indices[j]; 
            int col = i;           
            double val = values[j];
            A_trans.InsertGlobalValues(row, 1, &val, &col);
          }
        
        }
      }
      A_trans.FillComplete(single_core_row_A, single_core_col_A);
      A_ptr = std::make_shared<Epetra_CrsMatrix>(A_trans);
    }
    else{
      A_ptr = std::make_shared<Epetra_CrsMatrix>(A_temp);
    }
  
    return *A_ptr;
}

Epetra_CrsMatrix transpose_matrix_on_single_core(const Epetra_CrsMatrix &A, const bool is_input_A_matrix_transposed = false){
  // Gather Matrix Information
  const Epetra_Map& rowMap = A.RowMap();
  const Epetra_Map& colMap = A.ColMap();
  const int numMyRows = A.NumMyRows();
  
  const int max_num_elements_in_rows = A.Graph().GlobalMaxNumIndices();
  std::vector<double> values(max_num_elements_in_rows);
  std::vector<int> indices(max_num_elements_in_rows);
  
  std::shared_ptr<Epetra_CrsMatrix> A_ptr;
  if (is_input_A_matrix_transposed) {
    // Transpose mmatrix
    Epetra_CrsMatrix A_trans(Epetra_DataAccess::Copy, A.ColMap(), A.NumGlobalRows());
    for (int i = 0; i < numMyRows; ++i) {
      int numEntries;

      A.ExtractMyRowCopy(i, max_num_elements_in_rows, numEntries, values.data(), indices.data());

      int globalRow = rowMap.GID(i);
      for (int j = 0; j < numEntries; ++j) {
          int globalCol = colMap.GID(indices[j]);
          double val = values[j];

          A_trans.InsertGlobalValues(globalCol, 1, &val, &globalRow);
      }
  }
    A_trans.FillComplete(A.RowMap(), A.ColMap());
    A_ptr = std::make_shared<Epetra_CrsMatrix>(A_trans);
  }
  else{
    A_ptr = std::make_shared<Epetra_CrsMatrix>(A);
  }

  return *A_ptr;
}

Epetra_Vector copy_vector_to_all_cores(const Epetra_Vector &b){
    // Gather Vector Information
    const Epetra_SerialComm sComm;
    const int b_size = b.GlobalLength();
    // Create new map for one core and gather old map
    Epetra_Map single_core_b (b_size, b_size, 0, sComm);
    Epetra_BlockMap old_map_b = b.Map();
    // Create Epetra_importer object
    Epetra_Import b_importer(single_core_b, old_map_b);
    // Create new b vector
    Epetra_Vector b_temp (single_core_b); 
    // Load the data from vector b (Multi core) into b_temp (Single core)
    b_temp.Import(b, b_importer, Epetra_CombineMode::Insert);
    return b_temp;
}

Epetra_CrsMatrix copy_matrix_to_all_cores(const Epetra_CrsMatrix &A){
    // Gather Matrix Information
    const int A_rows = A.NumGlobalRows();
    const int A_cols = A.NumGlobalCols();

    // Create new maps for one core and gather old maps
    const Epetra_SerialComm sComm;
    Epetra_Map single_core_row_A (A_rows, A_rows, 0 , sComm);
    Epetra_Map single_core_col_A (A_cols, A_cols, 0 , sComm);
    Epetra_Map old_row_map_A = A.RowMap();
    Epetra_Map old_col_map_A = A.DomainMap();

    // Create Epetra_importer object
    Epetra_Import A_importer(single_core_row_A,old_row_map_A);

    // Create new A matrix
    Epetra_CrsMatrix A_temp (Epetra_DataAccess::Copy, single_core_row_A, A_cols);
    // Load the data from matrix A (Multi core) into A_temp (Single core)
    A_temp.Import(A, A_importer, Epetra_CombineMode::Insert);
    A_temp.FillComplete(single_core_col_A,single_core_row_A);
    return A_temp;
}