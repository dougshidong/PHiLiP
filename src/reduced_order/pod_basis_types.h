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
protected:
    /// Constructor
    SpecificPOD(const Parameters::AllParameters *parameters_input);

    /// Destructor
    ~SpecificPOD() {}

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basisTranspose; ///< Transpose of pod_basis

public:
    void addPODBasisColumns(const std::vector<unsigned int> addColumns);

    virtual void removePODBasisColumns(const std::vector<unsigned int> removeColumns);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

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
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

/// Class for fine not in coarse POD basis
class FineNotInCoarsePOD : public SpecificPOD
{
public:
    /// Constructor
    FineNotInCoarsePOD(const Parameters::AllParameters *parameters_input);
    /// Destructor
    ~FineNotInCoarsePOD () {};

    void removePODBasisColumns(const std::vector<unsigned int> removeColumns) override;

private:
//const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
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
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
};

}
}

#endif
