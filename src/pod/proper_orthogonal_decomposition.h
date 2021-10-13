#ifndef __POD__
#define __POD__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Proper Orthogonal Decomposition reduced order modelling
class POD
{
public:

    int num_basis; ///< Number of basis functions to keep for the reduced order model
    std::unique_ptr<dealii::LAPACKFullMatrix<double>> svd_u; ///< U matrix output from SVD
    dealii::TrilinosWrappers::SparseMatrix pod_basis; ///< First num_basis columns of svd_u

    /// Constructor
    POD(int num_basis);

    /// Constructor not specifying number of basis functions
    POD();

    /// Destructor
    ~POD () {};

    /// Get full POD basis consisting of svd_u
    dealii::LAPACKFullMatrix<double> get_full_pod_basis();

    /// Get reduced POD basis consisting of the first num_basis columns of svd_u
    void build_reduced_pod_basis();

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif
