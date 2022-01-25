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
class SpecificPOD : public POD<dim>
{
protected:
    /// Constructor
    SpecificPOD(std::shared_ptr<DGBase<dim,double>> &_dg);

    /// Destructor
    ~SpecificPOD() {}

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basisTranspose; ///< Transpose of pod_basis

public:
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

    std::vector<unsigned int> fullBasisIndices;

};

/// Class for Coarse POD basis
template <int dim>
class CoarsePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    CoarsePOD(std::shared_ptr<DGBase<dim,double>> &_dg);
    /// Destructor
    ~CoarsePOD () {};
};

/// Class for fine not in coarse POD basis
template <int dim>
class FineNotInCoarsePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    FineNotInCoarsePOD(std::shared_ptr<DGBase<dim,double>> &_dg);
    /// Destructor
    ~FineNotInCoarsePOD () {};

    void removePODBasisColumns(const std::vector<unsigned int> removeColumns);
};

/// Class for fine POD basis
template <int dim>
class FinePOD : public SpecificPOD<dim>
{
public:
    /// Constructor
    FinePOD(std::shared_ptr<DGBase<dim,double>> &_dg);
    /// Destructor
    ~FinePOD () {};
};

}
}

#endif
