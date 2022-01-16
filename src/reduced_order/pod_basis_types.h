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

class SpecificPOD : public POD
{
private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basisTranspose; ///< Transpose of pod_basis

protected:
    /// Constructor
    SpecificPOD();

    /// Destructor
    ~SpecificPOD() {}

public:
    void updatePODBasis(std::vector<unsigned int> indices);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

    std::vector<unsigned int> getHighestErrorBasis(int numBasisToAdd, dealii::LinearAlgebra::distributed::Vector<double> dualWeightedResidual);

    std::vector<unsigned int> fullBasisIndices;
};

/// Class for Coarse POD basis
class CoarsePOD : public SpecificPOD
{
public:
    /// Constructor
    CoarsePOD(const Parameters::AllParameters *parameters_input);
    /// Destructor
    ~CoarsePOD () {};

private:
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

/// Class for fine not in coarse POD basis
class FineNotInCoarsePOD : public SpecificPOD
{
public:
    /// Constructor
    FineNotInCoarsePOD(const Parameters::AllParameters *parameters_input);
    /// Destructor
    ~FineNotInCoarsePOD () {};

private:
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

/// Class for fine POD basis
class FinePOD : public SpecificPOD
{
public:
    /// Constructor
    FinePOD(const Parameters::AllParameters *parameters_input);
    /// Destructor
    ~FinePOD () {};

private:
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

}
}

#endif
