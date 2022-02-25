#ifndef __POD_BASIS_TYPES__
#define __POD_BASIS_TYPES__

#include <fstream>
#include <iostream>
#include <filesystem>

#include "functional/functional.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis.h"
#include "linear_solver/linear_solver.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
/// Intermediary class that includes attributes common to all POD basis subtypes
class SpecificPOD : public POD<dim>
{
protected:
    /// Constructor
    SpecificPOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~SpecificPOD() {}

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis; ///< pod basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basisTranspose; ///< Transpose of pod_basis

public:

    /// Function to add columns (basis functions) to POD basis. Used when building basis and refining when doing POD adaptation
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    /// Function to return basisTranspose
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

    /// Vector to store which indicies of the full basis are present in this basis
    std::vector<unsigned int> fullBasisIndices;

};

/// Class for Coarse POD basis, derived from SpecificPOD
template <int dim>
class CoarsePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    CoarsePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~CoarsePOD () {};
};

/// Class for fine not in coarse POD basis, derived from SpecificPOD
template <int dim>
class FineNotInCoarsePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    FineNotInCoarsePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~FineNotInCoarsePOD () {};

    /// Removes columns of the basis, used during POD adaptation
    void removePODBasisColumns(const std::vector<unsigned int> removeColumns);
};

/// Class for fine POD basis, derived from SpecificPOD
template <int dim>
class FinePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    FinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~FinePOD () {};
};

}
}

#endif
