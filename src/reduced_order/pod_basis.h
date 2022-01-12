#ifndef __POD_BASIS__
#define __POD_BASIS__

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
    dealii::LAPACKFullMatrix<double> fullPODBasis; ///< U matrix output from SVD, full POD basis
    dealii::TrilinosWrappers::SparseMatrix pod_basis; ///< First num_basis columns of fullPODBasis
    dealii::TrilinosWrappers::SparseMatrix pod_basis_transpose; ///< Transpose of pod_basis

    /// Constructor
    POD(int num_basis);

    /// Constructor not specifying number of basis functions
    POD();

    /// Destructor
    ~POD () {};

    /// Get full POD basis consisting of fullPODBasis
    void getPODBasisFromSnapshots();

    void getSavedPODBasis();

    void saveFullPODBasisToFile();



    /// Get reduced POD basis consisting of the first num_basis columns of fullPODBasis
    void build_reduced_pod_basis();

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif
