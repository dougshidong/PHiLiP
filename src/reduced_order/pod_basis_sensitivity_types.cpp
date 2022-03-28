#include "pod_basis_sensitivity_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/******************************************************************
*                     EXTRAPOLATED POD
******************************************************************/
template <int dim>
ExtrapolatedPOD<dim>::ExtrapolatedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODSensitivity<dim>(dg_input)
        , extrapolatedBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.extrapolated_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void ExtrapolatedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Extrapolated POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), this->fullBasisIndices.size(),this->fullBasisIndices.size());

    double delta = this->all_parameters->reduced_order_param.extrapolated_parameter_delta;
    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]) + delta*(this->fullBasisSensitivity(rowIndices[i], this->fullBasisIndices[j])));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    extrapolatedBasis->reinit(basis_tmp);
    extrapolatedBasis->copy_from(basis_tmp);
    this->pcout << "Extrapolated POD basis updated..." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> ExtrapolatedPOD<dim>::getPODBasis() {
    return extrapolatedBasis;
}


/******************************************************************
*                     COARSE EXPANDED POD
******************************************************************/

template <int dim>
CoarseExpandedPOD<dim>::CoarseExpandedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODSensitivity<dim>(dg_input)
        , coarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    int numVectors = this->all_parameters->reduced_order_param.coarse_expanded_basis_dimension;

    std::vector<unsigned int> initialBasisIndices(numVectors);
    //Expanded basis dimension should be even, first half is from state basis, second half is from sensitivity basis

    std::iota(initialBasisIndices.begin(), initialBasisIndices.begin() + numVectors/2, 0); //Fill first half as 0,1,2...
    std::iota(initialBasisIndices.begin() + numVectors/2, initialBasisIndices.end(), this->fullPODBasis->n()); //Fill second half as n, n+1, n+2...

    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void CoarseExpandedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Coarse Expanded POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
        this->pcout << idx << std::endl;
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasisStateAndSensitivity(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    coarseBasis->reinit(basis_tmp);
    coarseBasis->copy_from(basis_tmp);
    this->pcout << "Coarse Expanded POD basis updated..." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> CoarseExpandedPOD<dim>::getPODBasis() {
    return coarseBasis;
}

template <int dim>
std::vector<unsigned int> CoarseExpandedPOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

/******************************************************************
*                FINE NOT IN COARSE EXPANDED POD
******************************************************************/

template <int dim>
FineNotInCoarseExpandedPOD<dim>::FineNotInCoarseExpandedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODSensitivity<dim>(dg_input)
        , fineNotInCoarseBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    int start = this->all_parameters->reduced_order_param.coarse_expanded_basis_dimension/2;
    int numVectors = this->all_parameters->reduced_order_param.fine_expanded_basis_dimension - this->all_parameters->reduced_order_param.coarse_expanded_basis_dimension;
    std::vector<unsigned int> initialBasisIndices(numVectors);

    std::iota(initialBasisIndices.begin(), initialBasisIndices.begin() + numVectors/2, start);
    std::iota(initialBasisIndices.begin() + numVectors/2, initialBasisIndices.end(), this->fullPODBasis->n() + start);

    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void FineNotInCoarseExpandedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Fine not in Coarse Expanded POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
        this->pcout << idx << std::endl;
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasisStateAndSensitivity(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->reinit(basis_tmp);
    fineNotInCoarseBasis->copy_from(basis_tmp);
    this->pcout << " Fine not in Coarse Expanded POD basis updated..." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineNotInCoarseExpandedPOD<dim>::getPODBasis() {
    return fineNotInCoarseBasis;
}

template <int dim>
std::vector<unsigned int> FineNotInCoarseExpandedPOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

template <int dim>
void FineNotInCoarseExpandedPOD<dim>::removePODBasisColumns(const std::vector<unsigned int> removeColumns) {
    this->pcout << "Removing columns from Fine not in Coarse basis..." << std::endl;

    for (unsigned int idx: removeColumns) {
        this->fullBasisIndices.erase(std::remove(fullBasisIndices.begin(), fullBasisIndices.end(), idx), fullBasisIndices.end());
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasisStateAndSensitivity(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineNotInCoarseBasis->reinit(basis_tmp);
    fineNotInCoarseBasis->copy_from(basis_tmp);
    this->pcout << "Fine not in Coarse basis updated..." << std::endl;
}

/******************************************************************
*                     FINE EXPANDED POD
******************************************************************/

template <int dim>
FineExpandedPOD<dim>::FineExpandedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : PODSensitivity<dim>(dg_input)
        , fineBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    int numVectors = this->all_parameters->reduced_order_param.fine_expanded_basis_dimension;
    std::vector<unsigned int> initialBasisIndices(numVectors);
    //Expanded basis dimension should be even, first half is from state basis, second half is from sensitivity basis

    std::iota(initialBasisIndices.begin(), initialBasisIndices.begin() + numVectors/2, 0); //Fill first half as 0,1,2...
    std::iota(initialBasisIndices.begin() + numVectors/2, initialBasisIndices.end(), this->fullPODBasis->n()); //Fill second half as n, n+1, n+2...

    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void FineExpandedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Fine Expanded POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
        this->pcout << idx << std::endl;
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),fullBasisIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasisStateAndSensitivity(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);

    fineBasis->reinit(basis_tmp);
    fineBasis->copy_from(basis_tmp);
    this->pcout << "Fine Expanded POD basis updated..." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> FineExpandedPOD<dim>::getPODBasis() {
    return fineBasis;
}

template <int dim>
std::vector<unsigned int> FineExpandedPOD<dim>::getFullBasisIndices() {
    return fullBasisIndices;
}

template class CoarseExpandedPOD <PHILIP_DIM>;
template class FineNotInCoarseExpandedPOD <PHILIP_DIM>;
template class FineExpandedPOD <PHILIP_DIM>;
template class ExtrapolatedPOD <PHILIP_DIM>;

}
}