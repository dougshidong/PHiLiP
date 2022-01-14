#include "coarse_pod_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

CoarsePOD::CoarsePOD(const Parameters::AllParameters *const parameters_input)
        : POD(), coarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>()),
          coarseBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>()),
          all_parameters(parameters_input), mpi_communicator(MPI_COMM_WORLD),
          pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
    coarseBasisDim = all_parameters->reduced_order_param.coarse_basis_dimension;
    buildCoarsePODBasis();
}


std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasis() {
    return coarseBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasisTranspose() {
    return coarseBasisTranspose;
}

void CoarsePOD::buildCoarsePODBasis() {
    std::vector<int> row_index_set(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    std::vector<int> column_index_set(coarseBasisDim);
    std::iota(std::begin(column_index_set), std::end(column_index_set), 0);

    dealii::TrilinosWrappers::SparseMatrix coarse_basis_tmp(fullPODBasisLAPACK.n_rows(), coarseBasisDim,
                                                            coarseBasisDim);
    dealii::TrilinosWrappers::SparseMatrix coarse_basis_transpose_tmp(coarseBasisDim, fullPODBasisLAPACK.n_rows(),
                                                                      fullPODBasisLAPACK.n_rows());

    for (int i: row_index_set) {
        for (int j: column_index_set) {
            coarse_basis_tmp.set(i, j, fullPODBasisLAPACK(i, j));
            coarse_basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(i, j));
        }
    }

    coarse_basis_tmp.compress(dealii::VectorOperation::insert);
    coarse_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    coarseBasis->reinit(coarse_basis_tmp);
    coarseBasis->copy_from(coarse_basis_tmp);
    coarseBasisTranspose->reinit(coarse_basis_transpose_tmp);
    coarseBasisTranspose->copy_from(coarse_basis_transpose_tmp);
}

void CoarsePOD::updateCoarsePODBasis(std::vector<unsigned int> column_index_set) {
    pcout << "Updating Coarse POD basis..." << std::endl;
    std::vector<unsigned int> row_index_set(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    coarseBasisDim = (int) column_index_set.size();

    dealii::TrilinosWrappers::SparseMatrix coarse_basis_tmp(row_index_set.size(), column_index_set.size(),
                                                            column_index_set.size());
    dealii::TrilinosWrappers::SparseMatrix coarse_basis_transpose_tmp(column_index_set.size(), row_index_set.size(),
                                                                      row_index_set.size());

    for (unsigned int i = 0; i < row_index_set.size(); i++)  {
        for (unsigned int j = 0; j < column_index_set.size(); j++) {
            coarse_basis_tmp.set(i, j, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
            coarse_basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
        }
    }

    coarse_basis_tmp.compress(dealii::VectorOperation::insert);
    coarse_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    coarseBasis->reinit(coarse_basis_tmp);
    coarseBasis->copy_from(coarse_basis_tmp);
    coarseBasisTranspose->reinit(coarse_basis_transpose_tmp);
    coarseBasisTranspose->copy_from(coarse_basis_transpose_tmp);
    pcout << "Coarse POD basis updated..." << std::endl;
}

}
}