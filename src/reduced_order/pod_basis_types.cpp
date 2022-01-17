#include "pod_basis_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

SpecificPOD::SpecificPOD()
        : POD(), basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>()),
          basisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
          {}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificPOD::getPODBasis() {
    return basis;
}

std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificPOD::getPODBasisTranspose() {
    return basisTranspose;
}

void SpecificPOD::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(fullBasisIndices.size(), rowIndices.size(),
                                                               rowIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);
    basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    basis->reinit(basis_tmp);
    basis->copy_from(basis_tmp);
    basisTranspose->reinit(basis_transpose_tmp);
    basisTranspose->copy_from(basis_transpose_tmp);
    this->pcout << "POD basis updated..." << std::endl;
}

void SpecificPOD::removePODBasisColumns(const std::vector<unsigned int> /*removeColumns*/) {
    pcout << "Keeping all basis functions in the basis!" << std::endl;
}

CoarsePOD::CoarsePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

FinePOD::FinePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.fine_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

FineNotInCoarsePOD::FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.fine_basis_dimension -
                                               all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices),
              all_parameters->reduced_order_param.coarse_basis_dimension);
    addPODBasisColumns(initialBasisIndices);
}

void FineNotInCoarsePOD::removePODBasisColumns(const std::vector<unsigned int> removeColumns) {
    this->pcout << "Updating POD basis..." << std::endl;

    for (unsigned int idx: removeColumns) {
        fullBasisIndices.erase(std::remove(fullBasisIndices.begin(), fullBasisIndices.end(), idx), fullBasisIndices.end());
    }

    std::vector<unsigned int> rowIndices(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(fullBasisIndices.size(), rowIndices.size(),
                                                               rowIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);
    basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    this->basis->reinit(basis_tmp);
    this->basis->copy_from(basis_tmp);
    this->basisTranspose->reinit(basis_transpose_tmp);
    this->basisTranspose->copy_from(basis_transpose_tmp);
    this->pcout << "POD basis updated..." << std::endl;
}

}
}