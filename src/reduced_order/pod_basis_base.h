#ifndef __POD_BASIS_INTERFACE__
#define __POD_BASIS_INTERFACE__

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Interface for POD
template <int dim>
class PODBase
{
public:
    /// Virtual destructor
    virtual ~PODBase() = default;

    /// Function to return basis
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() = 0;

    /// Function to return reference state
    virtual dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() = 0;

    /// Function to return snapshot matrix
    virtual MatrixXd getSnapshotMatrix() = 0;
};

}
}


#endif