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
#include "pod_interfaces.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

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
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis();

    void addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot);

    void computeBasis();

    std::vector<dealii::LinearAlgebra::ReadWriteVector<double>> snapshotVectors;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    dealii::LAPACKFullMatrix<double> massMatrix;

    dealii::LAPACKFullMatrix<double> fullBasis;

};

}
}

#endif