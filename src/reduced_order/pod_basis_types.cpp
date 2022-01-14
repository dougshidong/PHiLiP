#include "pod_basis_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

CoarsePOD::CoarsePOD(const Parameters::AllParameters *const parameters_input)
    : POD()
    , coarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
    , coarseBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
    , all_parameters(parameters_input)
{
std::vector<unsigned int> column_index_set(all_parameters->reduced_order_param.coarse_basis_dimension);
std::iota(std::begin(column_index_set), std::end(column_index_set), 0);
updateCoarsePODBasis(column_index_set);
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasis() {
return coarseBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarsePOD::getPODBasisTranspose() {
return coarseBasisTranspose;
}

void CoarsePOD::updateCoarsePODBasis(std::vector<unsigned int> column_index_set) {
this->pcout << "Updating Coarse POD basis..." << std::endl;
std::vector<unsigned int> row_index_set(fullPODBasisLAPACK.n_rows());
std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

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
this->pcout << "Coarse POD basis updated..." << std::endl;
}

FinePOD::FinePOD(const Parameters::AllParameters *const parameters_input)
        : POD()
        , fineBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , fineBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , all_parameters(parameters_input)
{
    std::vector<unsigned int> column_index_set(all_parameters->reduced_order_param.fine_basis_dimension);
    std::iota(std::begin(column_index_set), std::end(column_index_set), 0);
    updateFinePODBasis(column_index_set);
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FinePOD::getPODBasis() {
    return fineBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FinePOD::getPODBasisTranspose() {
    return fineBasisTranspose;
}

void FinePOD::updateFinePODBasis(std::vector<unsigned int> column_index_set) {
    this->pcout << "Updating Fine POD basis..." << std::endl;
    std::vector<unsigned int> row_index_set(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    dealii::TrilinosWrappers::SparseMatrix fine_basis_tmp(row_index_set.size(), column_index_set.size(),
                                                          column_index_set.size());
    dealii::TrilinosWrappers::SparseMatrix fine_basis_transpose_tmp(column_index_set.size(), row_index_set.size(),
                                                                    row_index_set.size());

    for (unsigned int i = 0; i < row_index_set.size(); i++)  {
        for (unsigned int j = 0; j < column_index_set.size(); j++) {
            fine_basis_tmp.set(i, j, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
            fine_basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
        }
    }

    fine_basis_tmp.compress(dealii::VectorOperation::insert);
    fine_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    fineBasis->reinit(fine_basis_tmp);
    fineBasis->copy_from(fine_basis_tmp);
    fineBasisTranspose->reinit(fine_basis_transpose_tmp);
    fineBasisTranspose->copy_from(fine_basis_transpose_tmp);
    this->pcout << "Fine POD basis updated..." << std::endl;
}

FineNotInCoarsePOD::FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input)
        : POD()
        , fineNotInCoarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , fineNotInCoarseBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , all_parameters(parameters_input)
{
    std::vector<unsigned int> column_index_set(all_parameters->reduced_order_param.fine_basis_dimension - all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(column_index_set), std::end(column_index_set), all_parameters->reduced_order_param.coarse_basis_dimension);
    updateFineNotInCoarsePODBasis(column_index_set);
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarsePOD::getPODBasis() {
    return fineNotInCoarseBasis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarsePOD::getPODBasisTranspose() {
    return fineNotInCoarseBasisTranspose;
}

void FineNotInCoarsePOD::updateFineNotInCoarsePODBasis(std::vector<unsigned int> column_index_set) {
    this->pcout << "Updating fine not in coarse POD basis..." << std::endl;

    std::vector<unsigned int> row_index_set(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set), 0);

    dealii::TrilinosWrappers::SparseMatrix fine_not_in_coarse_basis_tmp(row_index_set.size(), column_index_set.size(),
                                                                        column_index_set.size());
    dealii::TrilinosWrappers::SparseMatrix fine_not_in_coarse_basis_transpose_tmp(column_index_set.size(), row_index_set.size(),
                                                                                  row_index_set.size());

    for (unsigned int i = 0; i < row_index_set.size(); i++)  {
        for (unsigned int j = 0; j < column_index_set.size(); j++) {
            fine_not_in_coarse_basis_tmp.set(i, j, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
            fine_not_in_coarse_basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(row_index_set[i], column_index_set[j]));
        }
    }

    fine_not_in_coarse_basis_tmp.compress(dealii::VectorOperation::insert);
    fine_not_in_coarse_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->reinit(fine_not_in_coarse_basis_tmp);
    fineNotInCoarseBasis->copy_from(fine_not_in_coarse_basis_tmp);
    fineNotInCoarseBasisTranspose->reinit(fine_not_in_coarse_basis_transpose_tmp);
    fineNotInCoarseBasisTranspose->copy_from(fine_not_in_coarse_basis_transpose_tmp);
    this->pcout << "Fine not in coarse POD basis updated..." << std::endl;
}

}
}