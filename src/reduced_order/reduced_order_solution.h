#ifndef __REDUCED_ORDER_SOLUTION__
#define __REDUCED_ORDER_SOLUTION__

#include "functional/functional.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_basis_base.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class to hold information about the reduced-order solution
template<int dim, int nstate>
class ROMSolution
{
public:
    /// Constructor
    ROMSolution(std::shared_ptr<DGBase<dim,double>> &dg_input, std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix_transpose, Functional<dim,nstate,double> &functional_input, std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis);

    /// Destructor
    ~ROMSolution () {};

    /// Stores system matrix transpose
    const std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix_transpose;

    /// Stores residual
    const dealii::LinearAlgebra::distributed::Vector<double> right_hand_side;

    /// Stores POD basis on which solution was computed
    const std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    /// Stores functional value
    const double functional_value;

    /// Stores gradient
    const dealii::LinearAlgebra::distributed::Vector<double> gradient;

};

}
}


#endif