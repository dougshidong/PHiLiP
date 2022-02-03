#include "pod_basis_types.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
SpecificPOD<dim>::SpecificPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : POD<dim>(dg_input)
        , basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , basisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
          {}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificPOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SpecificPOD<dim>::getPODBasisTranspose() {
    return basisTranspose;
}

template <int dim>
void SpecificPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> addColumns) {
    this->pcout << "Updating POD basis..." << std::endl;

    for (unsigned int idx: addColumns) {
        fullBasisIndices.push_back(idx);
    }

    std::vector<unsigned int> rowIndices(this->fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), fullBasisIndices.size(),
                                                     fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(fullBasisIndices.size(), rowIndices.size(),
                                                               rowIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, this->fullPODBasisLAPACK(rowIndices[i], fullBasisIndices[j]));
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

template <int dim>
CoarsePOD<dim>::CoarsePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SpecificPOD<dim>(dg_input)
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    this->addPODBasisColumns(initialBasisIndices);
}

template <int dim>
FinePOD<dim>::FinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SpecificPOD<dim>(dg_input)
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.fine_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices), 0);
    this->addPODBasisColumns(initialBasisIndices);
}

template <int dim>
FineNotInCoarsePOD<dim>::FineNotInCoarsePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : SpecificPOD<dim>(dg_input)
{
    std::vector<unsigned int> initialBasisIndices(this->all_parameters->reduced_order_param.fine_basis_dimension -
                                               this->all_parameters->reduced_order_param.coarse_basis_dimension);
    std::iota(std::begin(initialBasisIndices), std::end(initialBasisIndices),
              this->all_parameters->reduced_order_param.coarse_basis_dimension);
    this->addPODBasisColumns(initialBasisIndices);
}

template <int dim>
void FineNotInCoarsePOD<dim>::removePODBasisColumns(const std::vector<unsigned int> removeColumns) {
    this->pcout << "Updating POD basis..." << std::endl;

    for (unsigned int idx: removeColumns) {
        this->fullBasisIndices.erase(std::remove(this->fullBasisIndices.begin(), this->fullBasisIndices.end(), idx), this->fullBasisIndices.end());
    }

    std::vector<unsigned int> rowIndices(this->fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(rowIndices), std::end(rowIndices), 0);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(rowIndices.size(), this->fullBasisIndices.size(),
                                                     this->fullBasisIndices.size());
    dealii::TrilinosWrappers::SparseMatrix basis_transpose_tmp(this->fullBasisIndices.size(), rowIndices.size(),
                                                               rowIndices.size());

    for (unsigned int i = 0; i < rowIndices.size(); i++) {
        for (unsigned int j = 0; j < this->fullBasisIndices.size(); j++) {
            basis_tmp.set(i, j, this->fullPODBasisLAPACK(rowIndices[i], this->fullBasisIndices[j]));
            basis_transpose_tmp.set(j, i, this->fullPODBasisLAPACK(rowIndices[i], this->fullBasisIndices[j]));
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

template class SpecificPOD <PHILIP_DIM>;
template class FinePOD <PHILIP_DIM>;
template class CoarsePOD <PHILIP_DIM>;
template class FineNotInCoarsePOD <PHILIP_DIM>;

}
}