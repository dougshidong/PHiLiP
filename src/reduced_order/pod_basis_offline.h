#ifndef __POD_BASIS_OFFLINE__
#define __POD_BASIS_OFFLINE__

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_basis_base.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Class for Offline Proper Orthogonal Decomposition basis. This class reads some previously computed snapshots stored as files and computes a POD basis.
template <int dim>
class OfflinePOD: public PODBase<dim>
{
public:
    /// Constructor
    OfflinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~OfflinePOD () {};

    ///Function to get POD basis for all derived classes
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    ///Function to get POD reference state
    dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() override;

    /// Read snapshots to build POD basis
    bool getPODBasisFromSnapshots();

    /// POD basis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    /// Reference state
    dealii::LinearAlgebra::ReadWriteVector<double> referenceState;

    /// dg needed for sparsity pattern of system matrix
    std::shared_ptr<DGBase<dim,double>> dg;

    /// LAPACKFullMatrix for nice printing
    dealii::LAPACKFullMatrix<double> fullBasis;

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

