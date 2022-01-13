#ifndef __COARSE_POD_BASIS__
#define __COARSE_POD_BASIS__

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
class CoarsePOD : public POD
{
public:
    int coarseBasisDim;

    /// Constructor
    CoarsePOD(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~CoarsePOD () {};

    /// Get reduced POD basis consisting of the first num_basis columns of fullPODBasis
    void buildCoarsePODBasis();

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasisTranspose() override;

private:
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> coarseBasis; ///< First num_basis columns of fullPODBasis
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> coarseBasisTranspose; ///< Transpose of pod_basis
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

}
}

#endif
