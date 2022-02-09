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
/// Intermediary class that includes attributes common to all POD basis subtypes
class SensitivityPOD : public POD<dim>
{
public:

    /// Constructor
    SensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~SensitivityPOD() {}

    bool getSensitivityPODBasisFromSnapshots();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sensitivityBasis; ///< sensitivity basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sensitivityBasisTranspose; ///< Transpose of sensitivity basis

public:

    void computeBasisSensitivity();

    dealii::Vector<double> computeModeSensitivity(int k);

    /// Function to build sensitivity POD basis
    void buildSensitivityPODBasis();

    /// Function to return basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    /// Function to return basisTranspose
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

    /// Vector to store which indices of the full basis are present in this basis
    std::vector<unsigned int> fullBasisIndices;

protected:
    dealii::LAPACKFullMatrix<double> fullBasisSensitivity;
private:

    dealii::LAPACKFullMatrix<double> sensitivitySnapshots;
    dealii::LAPACKFullMatrix<double> massWeightedSensitivitySnapshots;
    dealii::LAPACKFullMatrix<double> eigenvaluesSensitivity; ///< Matrix of singular value derivatives along the diagonal


};

}
}

#endif
