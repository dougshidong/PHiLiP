#ifndef __POD_BASIS_SENSITIVITY__
#define __POD_BASIS_SENSITIVITY__

#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis.h"
#include <deal.II/lac/householder.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
/// Sensitivity POD basis class
class SensitivityPOD : public POD<dim>
{
public:

    /// Constructor
    SensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~SensitivityPOD() {}

    /// Generate Sensitivity POD basis from snapshots
    bool getSensitivityPODBasisFromSnapshots();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sensitivityBasis; ///< sensitivity basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sensitivityBasisTranspose; ///< Transpose of sensitivity basis

public:

    /// Compute POD basis sensitivities
    void computeBasisSensitivity();

    /// Compute the sensitivity k-th POD basis mode
    dealii::Vector<double> computeModeSensitivity(int k);

    /// Function to build sensitivity POD basis
    void buildSensitivityPODBasis();

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    /// Function to return basisTranspose
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

protected:

    dealii::LAPACKFullMatrix<double> fullBasisSensitivity; ///< Full sensitivity basis
private:

    /// Get Sensitivity POD basis saved to text file
    bool getSavedSensitivityPODBasis();

    /// Save Sensitivity POD basis to text file
    void saveSensitivityPODBasisToFile();

    dealii::LAPACKFullMatrix<double> sensitivitySnapshots; ///< Matrix of sensitivity snapshots
    dealii::LAPACKFullMatrix<double> massWeightedSensitivitySnapshots; ///< Mass matrix weighted sensitivity snapshots
    dealii::LAPACKFullMatrix<double> eigenvaluesSensitivity; ///< Matrix of singular value derivatives along the diagonal


};

}
}

#endif
