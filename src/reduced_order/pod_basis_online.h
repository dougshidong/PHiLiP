#ifndef __POD_BASIS_ONLINE__
#define __POD_BASIS_ONLINE__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_interface.h"
#include <deal.II/lac/la_parallel_vector.h>
#include <Teuchos_DefaultMpiComm.hpp>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Class for full Proper Orthogonal Decomposition basis
template <int dim>
class OnlinePOD: public POD<dim>
{
public:
    /// Constructor
    OnlinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    ~OnlinePOD () {};

    ///Function to get POD basis for all derived classes
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    ///Function to get POD reference state
    dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() override;

    void addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot);

    void computeBasis();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    dealii::LinearAlgebra::ReadWriteVector<double> referenceState;

    std::shared_ptr<DGBase<dim,double>> dg;

    dealii::LAPACKFullMatrix<double> dealiiSnapshotMatrix;

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