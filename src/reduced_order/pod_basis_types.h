#ifndef __POD_BASIS_TYPES__
#define __POD_BASIS_TYPES__

#include <fstream>
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include "dg/dg.h"
#include "reduced_order/pod_state_base.h"
#include "linear_solver/linear_solver.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "ode_solver/ode_solver_factory.h"
#include "pod_interfaces.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Coarse state POD basis, derived from PODState and implementing CoarseBasis
template <int dim>
class CoarseStatePOD : public PODState<dim>, public CoarseBasis<dim>
{
private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> coarseBasis; ///< pod basis
    std::vector<unsigned int> fullBasisIndices; ///< Vector to store which indicies of the full basis are present in this basis

public:
    /// Constructor
    CoarseStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~CoarseStatePOD () {};

    /// Add columns to the basis
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    /// Return vector storing which indices of the full basis are present in this basis
    std::vector<unsigned int> getFullBasisIndices();
};

/// Class for Fine not in Coarse state POD basis, derived from PODState and implementing FineNotInCoarseBasis
template <int dim>
class FineNotInCoarseStatePOD : public PODState<dim>, public FineNotInCoarseBasis<dim>
{
private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineNotInCoarseBasis; ///< pod basis
    std::vector<unsigned int> fullBasisIndices; ///< Vector to store which indicies of the full basis are present in this basis

public:
    /// Constructor
    FineNotInCoarseStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~FineNotInCoarseStatePOD () {};

    /// Add columns to the basis
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    /// Removes columns of the basis, used during POD adaptation
    void removePODBasisColumns(const std::vector<unsigned int> removeColumns);

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    /// Return vector storing which indices of the full basis are present in this basis
    std::vector<unsigned int> getFullBasisIndices();
};

/// Class for Fine state POD basis, derived from PODState and implementing FineBasis
template <int dim>
class FineStatePOD : public PODState<dim>, public FineBasis<dim>
{
private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineBasis; ///< pod basis
    std::vector<unsigned int> fullBasisIndices; ///< Vector to store which indicies of the full basis are present in this basis

public:
    /// Constructor
    FineStatePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);
    /// Destructor
    ~FineStatePOD () {};

    /// Add columns to the basis
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    /// Return vector storing which indices of the full basis are present in this basis
    std::vector<unsigned int> getFullBasisIndices();
};

}
}

#endif