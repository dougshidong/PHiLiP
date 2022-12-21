#ifndef __POD_BASIS_ONLINE__
#define __POD_BASIS_ONLINE__

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_basis_base.h"
#include <deal.II/lac/la_parallel_vector.h>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Class for Online Proper Orthogonal Decomposition basis. This class takes snapshots on the fly and computes a POD basis for use in adaptive sampling.
template <int dim>
class OnlinePOD: public PODBase<dim>
{
public:
    /// Constructor
    OnlinePOD(std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _system_matrix);

    /// Destructor
    ~OnlinePOD () {};

    ///Function to get POD basis for all derived classes
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    ///Function to get POD reference state
    dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() override;

    /// Add snapshot
    void addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot);

    /// Compute new POD basis from snapshots
    void computeBasis();

    /// POD basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    /// Reference state
    dealii::LinearAlgebra::ReadWriteVector<double> referenceState;

    /// For sparsity pattern of system matrix
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix;

    /// LAPACK matrix of snapshots for nice printing
    dealii::LAPACKFullMatrix<double> dealiiSnapshotMatrix;

    /// Matrix containing snapshots
    MatrixXd snapshotMatrix;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;


};

}
}

#endif
