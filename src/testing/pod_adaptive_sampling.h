#ifndef __POD_ADAPTIVE_SAMPLING__
#define __POD_ADAPTIVE_SAMPLING__

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
#include "reduced_order/pod_basis_online.h"
#include "reduced_order/reduced_order_solution.h"
#include "reduced_order/full_order_solution.h"
#include "linear_solver/linear_solver.h"
#include "testing/flow_solver.h"
#include "reduced_order/rom_test_location.h"
#include <deal.II/lac/householder.h>
#include <cmath>
#include <iostream>
#include <deal.II/base/function_lib.h>
#include "testing/reduced_order_pod_adaptation.h" //For burgers rewienski functional
#include <Eigen/Dense>
#include "reduced_order/delaunay.h"
#include "reduced_order/rbf_interpolation.h"
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"

namespace PHiLiP {
namespace Tests {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVector2d;
using Eigen::VectorXd;

//Refer to https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
template<typename T>
struct matrix_hash : std::unary_function<T, long> {
long operator()(T const& matrix) const {
    long seed = 0;
    for (long i = 0; i < matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
};

/// Class to hold information about the reduced-order solution
template <int dim, int nstate>
class AdaptiveSampling: public TestsBase
{
public:
    /// Constructor
    AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~AdaptiveSampling() {};

    mutable std::unordered_map<RowVector2d, ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>, matrix_hash<RowVector2d>> rom_locations;

    mutable MatrixXd snapshot_parameters;

    mutable double max_error;


    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    /// Run test
    int run_test () const override;

    void placeInitialSnapshots() const;

    void placeInitialROMs() const;

    void placeTriangulationROMs(ProperOrthogonalDecomposition::Delaunay delaunay) const;

    RowVector2d getMaxErrorROM() const;

    std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> solveSnapshotFOM(RowVector2d parameter) const;

    std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(RowVector2d parameter) const;

    Parameters::AllParameters reinitParams(RowVector2d parameter) const;

    void outputErrors(int iteration) const;
};

}
}


#endif