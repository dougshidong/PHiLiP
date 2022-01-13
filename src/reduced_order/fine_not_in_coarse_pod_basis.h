#ifndef __FINE_NOT_IN_COARSE_POD_BASIS__
#define __FINE_NOT_IN_COARSE_POD_BASIS__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Coarse POD basis
class FineNotInCoarsePOD : public POD
{
public:
    int fineNotInCoarseBasisDim;

    /// Constructor
    FineNotInCoarsePOD(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~FineNotInCoarsePOD () {};

    /// Get reduced POD basis consisting of the first num_basis columns of fullPODBasisLAPACK
    void buildFineNotInCoarsePODBasis();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineNotInCoarseBasis; ///< First num_basis columns of fullPODBasisLAPACK
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> fineNotInCoarseBasisTranspose; ///< Transpose of pod_basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

}
}

#endif
