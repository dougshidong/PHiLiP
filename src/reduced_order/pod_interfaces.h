#ifndef __POD_BASIS_INTERFACE__
#define __POD_BASIS_INTERFACE__

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Interface for POD
template <int dim>
class POD
{
public:
    /// Virtual destructor
    virtual ~POD() {};

    /// Function to return basis
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() = 0;

    virtual dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() {return dealii::LinearAlgebra::ReadWriteVector<double>(0);};
};

/// Interface for coarse basis
template <int dim>
class CoarseBasis : public POD<dim>
{
public:
    /// Virtual destructor
    virtual ~CoarseBasis () {};

    /// Add columns to the basis
    virtual void addPODBasisColumns(std::vector<unsigned int> addColumns) = 0;

    /// Return vector storing which indices of the full basis are present in this basis
    virtual std::vector<unsigned int> getFullBasisIndices() = 0;
};

/// Interface for fine not in coarse basis
template <int dim>
class FineNotInCoarseBasis : public POD<dim>
{
public:
    /// Virtual destructor
    virtual ~FineNotInCoarseBasis () {};

    /// Removes columns of the basis, used during POD adaptation
    virtual void removePODBasisColumns(std::vector<unsigned int> removeColumns) = 0;

    /// Add columns to the basis
    virtual void addPODBasisColumns(std::vector<unsigned int> addColumns) = 0;

    /// Return vector storing which indices of the full basis are present in this basis
    virtual std::vector<unsigned int> getFullBasisIndices() = 0;
};

/// Interface for fine basis
template <int dim>
class FineBasis : public POD<dim>
{
public:
    /// Virtual destructor
    virtual ~FineBasis () {};

    /// Add columns to the basis
    virtual void addPODBasisColumns(std::vector<unsigned int> addColumns) = 0;

    /// Return vector storing which indices of the full basis are present in this basis
    virtual std::vector<unsigned int> getFullBasisIndices() = 0;
};

}
}


#endif