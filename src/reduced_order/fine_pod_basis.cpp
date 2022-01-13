#include "fine_pod_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

FinePOD::FinePOD(const Parameters::AllParameters *const parameters_input)
        : POD()
        , all_parameters(parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    fineBasisDim = all_parameters->reduced_order_param.fine_basis_dimension;
    buildFinePODBasis();
}


std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FinePOD::getPODBasis() {
    return fineBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FinePOD::getPODBasisTranspose() {
    return fineBasisTranspose;
}

void FinePOD::buildFinePODBasis() {
    std::vector<int> row_index_set(fullPODBasis.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    std::vector<int> column_index_set(fineBasisDim);
    std::iota(std::begin(column_index_set), std::end(column_index_set), 0);

    dealii::TrilinosWrappers::SparseMatrix fine_basis_tmp(fullPODBasis.n_rows(), fineBasisDim, fineBasisDim);
    dealii::TrilinosWrappers::SparseMatrix fine_basis_transpose_tmp(fineBasisDim, fullPODBasis.n_rows(), fullPODBasis.n_rows());

    for (int i: row_index_set) {
        for (int j: column_index_set) {
            fine_basis_tmp.set(i, j, fullPODBasis(i, j));
            fine_basis_transpose_tmp.set(j, i, fullPODBasis(i, j));
        }
    }

    fine_basis_tmp.compress(dealii::VectorOperation::insert);
    fine_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    fineBasis->copy_from(fine_basis_tmp);
    fineBasisTranspose->copy_from(fine_basis_transpose_tmp);
}

}
}
