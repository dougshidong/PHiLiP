#include "pod_basis_sensitivity_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
SpecificSensitivityPOD<dim>::SpecificSensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SensitivityPOD<dim>(dg_input)
        , basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , basisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificSensitivityPOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificSensitivityPOD<dim>::getPODBasisTranspose() {
    return basisTranspose;
}


template <int dim>
ExpandedPOD<dim>::ExpandedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SpecificSensitivityPOD<dim>(dg_input)
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.expanded_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void ExpandedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Expanded POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        this->fullBasisIndices.push_back(idx);
        this->fullSensitivityBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), 2*this->fullBasisIndices.size(),2*this->fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(2*this->fullBasisIndices.size(), rowIndices.size(), rowIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            this->pcout << i << " " << j << std::endl;
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]));
        }
        int offset = this->fullBasisIndices.size();
        for (unsigned int j = offset; j < this->fullSensitivityBasisIndices.size() + offset; j++) {
            this->pcout << i << " " << j << std::endl;
            basis_tmp.set(i, j, this->fullBasisSensitivity(rowIndices[i], this->fullBasisIndices[j-offset]));
            basis_transpose_tmp.set(j, i, this->fullBasisSensitivity(rowIndices[i], this->fullBasisIndices[j-offset]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);
    basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    this->basis->reinit(basis_tmp);
    this->basis->copy_from(basis_tmp);
    this->basisTranspose->reinit(basis_tmp);
    this->basisTranspose->copy_from(basis_transpose_tmp);
    this->pcout << "Expanded POD basis updated..." << std::endl;
}

template <int dim>
ExtrapolatedPOD<dim>::ExtrapolatedPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SpecificSensitivityPOD<dim>(dg_input)
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.extrapolated_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void ExtrapolatedPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating Extrapolated POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        this->fullBasisIndices.push_back(idx);
        this->fullSensitivityBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullBasis.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), this->fullBasisIndices.size(),this->fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(this->fullBasisIndices.size(), rowIndices.size(), rowIndices.size());

    double delta = this->all_parameters->reduced_order_param.extrapolated_parameter_delta;
    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]) + delta*this->fullBasisSensitivity(rowIndices[i], this->fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, this->fullBasis(rowIndices[i], this->fullBasisIndices[j]) + delta*this->fullBasisSensitivity(rowIndices[i], this->fullBasisIndices[j]));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::insert);
    basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    this->basis->reinit(basis_tmp);
    this->basis->copy_from(basis_tmp);
    this->basisTranspose->reinit(basis_transpose_tmp);
    this->basisTranspose->copy_from(basis_transpose_tmp);
    this->pcout << "Extrapolated POD basis updated..." << std::endl;
}



template class SpecificSensitivityPOD <PHILIP_DIM>;
template class ExpandedPOD <PHILIP_DIM>;
template class ExtrapolatedPOD <PHILIP_DIM>;

}
}