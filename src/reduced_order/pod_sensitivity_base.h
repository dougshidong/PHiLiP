#ifndef __POD_SENSITIVITY_BASE__
#define __POD_SENSITIVITY_BASE__

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
#include "reduced_order/pod_state_base.h"
#include <deal.II/lac/householder.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
/// Sensitivity POD basis class
class PODSensitivity : public PODState<dim>
{
public:
    /// Constructor
    PODSensitivity(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~PODSensitivity() {}

    /// Generate Sensitivity POD basis from snapshots
    bool getSensitivityPODBasisFromSnapshots();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sensitivityBasis; ///< sensitivity basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> stateAndSensitivityBasis; ///< sensitivity basis

    /// Compute POD basis sensitivities
    void computeBasisSensitivity();

    /// Compute the sensitivity k-th POD basis mode
    dealii::Vector<double> computeModeSensitivity(int k);

    /// Function to build sensitivity POD basis
    void buildSensitivityPODBasis();

    /// Function to build state and sensitivity POD basis
    void buildStateAndSensitivityPODBasis();

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;


protected:
    dealii::LAPACKFullMatrix<double> fullBasisSensitivity; ///< Full sensitivity basis only
    dealii::LAPACKFullMatrix<double> fullBasisStateAndSensitivity; ///< Full state and sensitivity basis

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