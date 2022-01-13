#include "fine_not_in_coarse_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

FineNotInCoarsePOD::FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input)
        : POD(), all_parameters(parameters_input), mpi_communicator(MPI_COMM_WORLD),
          pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    fineNotInCoarseBasisDim = all_parameters->reduced_order_param.coarse_basis_dimension;
    buildFineNotInCoarsePODBasis();
}


std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarsePOD::getPODBasis() {
    return fineNotInCoarseBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarsePOD::getPODBasisTranspose() {
    return fineNotInCoarseBasisTranspose;
}
// IMPLEMENTATION NOT CORRECT YET
void FineNotInCoarsePOD::buildFineNotInCoarsePODBasis() {
    std::vector<int> row_index_set(fullPODBasis.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    std::vector<int> column_index_set(fineNotInCoarseBasisDim);
    std::iota(std::begin(column_index_set), std::end(column_index_set), 0);

    dealii::TrilinosWrappers::SparseMatrix fine_not_in_coarse_basis_tmp(fullPODBasis.n_rows(), fineNotInCoarseBasisDim, fineNotInCoarseBasisDim);
    dealii::TrilinosWrappers::SparseMatrix fine_not_in_coarse_basis_transpose_tmp(fineNotInCoarseBasisDim, fullPODBasis.n_rows(), fullPODBasis.n_rows());

    for (int i: row_index_set) {
        for (int j: column_index_set) {
            fine_not_in_coarse_basis_tmp.set(i, j, fullPODBasis(i, j));
            fine_not_in_coarse_basis_transpose_tmp.set(j, i, fullPODBasis(i, j));
        }
    }

        fine_not_in_coarse_basis_tmp.compress(dealii::VectorOperation::insert);
        fine_not_in_coarse_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->copy_from(fine_not_in_coarse_basis_tmp);
    fineNotInCoarseBasisTranspose->copy_from(fine_not_in_coarse_basis_transpose_tmp);
}

}
}