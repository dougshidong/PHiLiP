#ifndef __REDUCED_ORDER_SOLUTION__
#define __REDUCED_ORDER_SOLUTION__

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/vector_tools.h>

#include "dg/dg_base.hpp"
#include "functional/functional.h"
#include "parameters/all_parameters.h"
#include "pod_basis_base.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class to hold information about the reduced-order solution
template<int dim, int nstate>
class ROMSolution
{
public:
    /// Constructor
    ROMSolution(Parameters::AllParameters params, dealii::LinearAlgebra::distributed::Vector<double> _solution, dealii::LinearAlgebra::distributed::Vector<double> _gradient);

    /// Stores all parameters
    Parameters::AllParameters params;

    /// Stores solution
    dealii::LinearAlgebra::distributed::Vector<double> solution;

    /// Stores gradient
    dealii::LinearAlgebra::distributed::Vector<double> gradient;

};

}
}


#endif