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
#include "dg/dg.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for full Proper Orthogonal Decomposition basis
template <int dim>
class POD
{
public:
    /// Constructor
    POD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    /// Destructor
    virtual ~POD () {};

    ///Virtual function to get POD basis for all derived classes
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    ///Virtual function to get POD basis transpose for all derived classes
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose();

private:
    /// Get full POD basis consisting of fullPODBasisLAPACK
    bool getPODBasisFromSnapshots();

    /// Get POD basis saved to text file
    bool getSavedPODBasis();

    /// Save POD basis to text file
    void saveFullPODBasisToFile();

    /// Build POD basis consisting of the first num_basis columns of fullPODBasisLAPACK
    void buildPODBasis();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fullPODBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fullPODBasisTranspose; ///< Transpose of pod_basis
    std::shared_ptr<DGBase<dim,double>> dg; ///< Smart pointer to DGBase

protected:
    dealii::LAPACKFullMatrix<double> fullPODBasisLAPACK; ///< U matrix output from SVD, full POD basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif
