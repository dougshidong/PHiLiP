#include "coarse_pod_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

CoarsePOD::CoarsePOD(const Parameters::AllParameters *const parameters_input)
        : POD()
        , all_parameters(parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    coarseBasisDim = all_parameters->reduced_order_param.coarse_basis_dimension;
    buildCoarsePODBasis();
}


std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasis(){
    return coarseBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasisTranspose(){
    return coarseBasisTranspose;
}

void CoarsePOD::buildCoarsePODBasis() {
    std::vector<int> row_index_set(fullPODBasis.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(coarseBasisDim);
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix coarse_basis_tmp(fullPODBasis.n_rows(), coarseBasisDim, coarseBasisDim);
    dealii::TrilinosWrappers::SparseMatrix coarse_basis_transpose_tmp(coarseBasisDim, fullPODBasis.n_rows(), fullPODBasis.n_rows());

    for(int i : row_index_set){
        for(int j : column_index_set){
            coarse_basis_tmp.set(i, j, fullPODBasis(i, j));
            coarse_basis_transpose_tmp.set(j, i, fullPODBasis(i, j));
        }
    }

    coarse_basis_tmp.compress(dealii::VectorOperation::insert);
    coarse_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    coarseBasis->copy_from(coarse_basis_tmp);
    coarseBasisTranspose->copy_from(coarse_basis_transpose_tmp);
}

}
}