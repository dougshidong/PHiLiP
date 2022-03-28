#include "pod_basis_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/******************************************************************
*                     COARSE STATE POD
******************************************************************/

template <int dim>
CoarseStatePOD<dim>::CoarseStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODState<dim>(dg_input)
        , coarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarseStatePOD<dim>::getPODBasis() {
    return coarseBasis;
}

template <int dim>
std::vector<unsigned int> CoarseStatePOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

template <int dim>
void CoarseStatePOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Coarse POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    coarseBasis->reinit(basis_tmp);
    coarseBasis->copy_from(basis_tmp);
    this->pcout << "Coarse POD basis updated..." << std::endl;
}

/******************************************************************
*                FINE NOT IN COARSE STATE POD
******************************************************************/

template <int dim>
FineNotInCoarseStatePOD<dim>::FineNotInCoarseStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODState<dim>(dg_input)
        , fineNotInCoarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.fine_basis_dimension -
                                                  this->all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices),
              this->all_parameters->reduced_order_param.coarse_basis_dimension);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarseStatePOD<dim>::getPODBasis() {
    return fineNotInCoarseBasis;
}

template <int dim>
std::vector<unsigned int> FineNotInCoarseStatePOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

template <int dim>
void FineNotInCoarseStatePOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Fine not in Coarse POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->reinit(basis_tmp);
    fineNotInCoarseBasis->copy_from(basis_tmp);
    this->pcout << "Fine not in Coarse POD basis updated..." << std::endl;
}

template <int dim>
void FineNotInCoarseStatePOD<dim>::removePODBasisColumns(const std::vector<unsigned int> removeColumns) {
    this->pcout << "Updating Fine not in Coarse POD basis..." << std::endl;

    for (unsigned int idx: removeColumns) {
        this->fullBasisIndices.erase(std::remove(this->fullBasisIndices.begin(), this->fullBasisIndices.end(), idx), this->fullBasisIndices.end());
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), this->fullBasisIndices.size(),
                                                     this->fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->reinit(basis_tmp);
    fineNotInCoarseBasis->copy_from(basis_tmp);
    this->pcout << "Fine not in Coarse POD basis updated..." << std::endl;
}

/******************************************************************
*                         FINE STATE POD
******************************************************************/

template <int dim>
FineStatePOD<dim>::FineStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODState<dim>(dg_input)
        , fineBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.fine_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineStatePOD<dim>::getPODBasis() {
    return fineBasis;
}

template <int dim>
std::vector<unsigned int> FineStatePOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

template <int dim>
void FineStatePOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Coarse POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineBasis->reinit(basis_tmp);
    fineBasis->copy_from(basis_tmp);
    this->pcout << "Coarse POD basis updated..." << std::endl;
}

template class FineStatePOD <PHILIP_DIM>;
template class CoarseStatePOD <PHILIP_DIM>;
template class FineNotInCoarseStatePOD <PHILIP_DIM>;

}
}