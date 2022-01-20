#ifndef __POD_BASIS__
#define __POD_BASIS__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Proper Orthogonal Decomposition reduced order modelling
class POD
{
public:
    dealii::LAPACKFullMatrix<double> fullPODBasisLAPACK; ///< U matrix output from SVD, full POD basis

    /// Constructor
    POD();

    /// Destructor
    virtual ~POD () {};

    /// Get full POD basis consisting of fullPODBasisLAPACK
    bool getPODBasisFromSnapshots();

    bool getSavedPODBasis();

    void saveFullPODBasisToFile();

    /// Get reduced POD basis consisting of the first num_basis columns of fullPODBasisLAPACK
    void buildPODBasis();

    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose();

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fullPODBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fullPODBasisTranspose; ///< Transpose of pod_basis

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif