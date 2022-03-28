#ifndef __FULL_ORDER_SOLUTION__
#define __FULL_ORDER_SOLUTION__

#include <fstream>
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod_interfaces.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class to hold information about the reduced-order solution
template <int dim, int nstate>
class FOMSolution
{
public:
    /// Constructor
    FOMSolution(std::shared_ptr<DGBase<dim,double>> &dg_input, Functional<dim,nstate,double> &functional_input, double sensitivity);

    /// Destructor
    ~FOMSolution () {};

    const dealii::LinearAlgebra::distributed::Vector<double> state;

    const double sensitivity;

    const double functional_value;
};

}
}


#endif