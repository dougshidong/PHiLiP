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

void SpecificPOD::updatePODBasis(const std::vector<unsigned int> newColumns) {
    this->pcout << "Updating POD basis..." << std::endl;

    for (unsigned int idx: newColumns) {
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

std::vector<unsigned int> SpecificPOD::getHighestErrorBasis(int numBasisToAdd, dealii::LinearAlgebra::distributed::Vector<double> dualWeightedResidual){
    //Generate vector of indices
    std::vector<unsigned int> reducedDualWeightedResidualIndices(dualWeightedResidual.size(), 0);
    for (unsigned int i = 0 ; i < dualWeightedResidual.size() ; i++) {
        reducedDualWeightedResidualIndices[i] = i;
    }

    //Sort indices based on reduced dual weighted residual
    std::sort (reducedDualWeightedResidualIndices.begin(), reducedDualWeightedResidualIndices.end(), [&](auto &a, auto &b) {return (dualWeightedResidual[a] > dualWeightedResidual[b]);});
    reducedDualWeightedResidualIndices.resize(numBasisToAdd);

    //Generate new indices to add to coarse basis
    std::vector<unsigned int> newIndices;

    for (unsigned int idx: reducedDualWeightedResidualIndices) {
        pcout << idx << std::endl;
        newIndices.push_back(fullBasisIndices[idx]);
    }

    return newIndices;
}

CoarsePOD::CoarsePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    updatePODBasis(initialBasisIndices);
}

FinePOD::FinePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.fine_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    updatePODBasis(initialBasisIndices);
}

FineNotInCoarsePOD::FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input)
        : SpecificPOD(), all_parameters(parameters_input) {
    std::vector<unsigned int> initialBasisIndices(all_parameters->reduced_order_param.fine_basis_dimension -
                                               all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices),
              all_parameters->reduced_order_param.coarse_basis_dimension);
    updatePODBasis(initialBasisIndices);
}

}
}