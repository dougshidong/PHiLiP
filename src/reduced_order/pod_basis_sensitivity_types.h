#ifndef __POD_BASIS_SENSITIVITY_TYPES__
#define __POD_BASIS_SENSITIVITY_TYPES__

#include <fstream>
#include <iostream>
#include <filesystem>

#include "functional/functional.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis.h"
#include "reduced_order/pod_basis_sensitivity.h"
#include "linear_solver/linear_solver.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "ode_solver/ode_solver_factory.h"
#include <algorithm>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template<int dim>
/// Intermediary class that includes attributes common to all POD basis subtypes
class SpecificSensitivityPOD : public SensitivityPOD<dim> {
protected:
/// Constructor
    SpecificSensitivityPOD(std::shared_ptr<DGBase<dim, double>> &dg_input);

/// Destructor
    ~SpecificSensitivityPOD() {}

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis; ///< pod basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basisTranspose; ///< Transpose of pod_basis

public:

/// Function to add columns (basis functions) to POD basis. Used when building basis and refining when doing POD adaptation
    virtual void addPODBasisColumns(const std::vector<unsigned int> addColumns) = 0;

/// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

/// Function to return basisTranspose
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

/// Vector to store which indicies of the full basis are present in this basis
    std::vector<unsigned int> fullBasisIndices;

    /// Vector to store which indicies of the sensitivity basis are present in this basis
    std::vector<unsigned int> fullSensitivityBasisIndices;

};

/// Class for Coarse POD basis, derived from SpecificPOD
template<int dim>
class ExpandedPOD : public SpecificSensitivityPOD<dim> {
public:
/// Constructor
    ExpandedPOD(std::shared_ptr<DGBase<dim, double>> &dg_input);

/// Destructor
    ~ExpandedPOD() {};

    void addPODBasisColumns(const std::vector<unsigned int> addColumns);
};

/// Class for fine not in coarse POD basis, derived from SpecificPOD
template<int dim>
class ExtrapolatedPOD : public SpecificSensitivityPOD<dim> {
public:
/// Constructor
    ExtrapolatedPOD(std::shared_ptr<DGBase<dim, double>> &dg_input);

/// Destructor
    ~ExtrapolatedPOD() {};

    void addPODBasisColumns(const std::vector<unsigned int> addColumns);
};

}
}

#endif
