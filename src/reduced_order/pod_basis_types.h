#ifndef __POD_BASIS_TYPES__
#define __POD_BASIS_TYPES__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Coarse POD basis
class CoarsePOD : public POD
{
public:
    /// Constructor
    CoarsePOD(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~CoarsePOD () {};

    void updateCoarsePODBasis(std::vector<unsigned int> indices);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> coarseBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> coarseBasisTranspose; ///< Transpose of pod_basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

/// Class for fine not in coarse POD basis
class FineNotInCoarsePOD : public POD
{
public:
    /// Constructor
    FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~FineNotInCoarsePOD () {};

    /// Get reduced POD basis consisting of the first num_basis columns of fullPODBasisLAPACK
    void updateFineNotInCoarsePODBasis(std::vector<unsigned int> indices);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineNotInCoarseBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineNotInCoarseBasisTranspose; ///< Transpose of pod_basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

};

/// Class for fine POD basis
class FinePOD : public POD
{
public:
    /// Constructor
    FinePOD(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~FinePOD () {};

    void updateFinePODBasis(std::vector<unsigned int> indices);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineBasisTranspose; ///< Transpose of pod_basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

}
}

#endif
