#ifndef __POD_ADAPTATION__
#define __POD_ADAPTATION__

#include <fstream>
#include <iostream>
#include <filesystem>

#include "functional/functional.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis_types.h"
#include <deal.II/base/function_parser.h>
#include "linear_solver/linear_solver.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

/// Class for Proper Orthogonal Decomposition reduced order modelling adaptation
/* Refer to "Output Error Estimation for Projection-Based Reduced Models" by Gary Collins, Krzysztof J. Fidkowski
and Carlos E. S. Cesnik, AIAA Aviation Forum 2019
*/

template <int dim, int nstate>
class PODAdaptation
{
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;

private:
    /// Functional
    Functional<dim,nstate,double> &functional;

    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;

    /// Pointer to all parameters
    const Parameters::AllParameters *const all_parameters;

    /// Smart pointer to coarse POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::CoarsePOD<dim>> coarsePOD;

    /// Smart pointer to fine POD basis
    std::unique_ptr<ProperOrthogonalDecomposition::FinePOD<dim>> finePOD;

    /// Smart pointer to fine not incoarse POD basis
    std::unique_ptr<ProperOrthogonalDecomposition::FineNotInCoarsePOD<dim>> fineNotInCoarsePOD;

    /// Smart pointer to ode_solver
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

    /// Linear solver parameters.
    Parameters::LinearSolverParam linear_solver_param;

    /// Dual-weighted residual
    DealiiVector dualWeightedResidual;

    /// Adaptation error
    double error;

public:
    /// Constructor
    PODAdaptation(std::shared_ptr<DGBase<dim,double>> &dg_input, Functional<dim,nstate,double> &functional_input);

    /// Destructor
    ~PODAdaptation () {};

    /// Compute reduced-order gradient
    void getReducedGradient(DealiiVector &reducedGradient);

    /// Apply reduced-order Jacobian transpose to solve for reduced-order adjoint
    void applyReducedJacobianTranspose(DealiiVector &reducedAdjoint, DealiiVector &reducedGradient);

    /// Simple adaptation algorithm
    void simplePODAdaptation();

    /// Progressive (iterative) adaptation algorithm
    void progressivePODAdaptation();

    /// Compute dual-weighted residual
    void getDualWeightedResidual();

    /// Determine which POD basis to add based on dual-weighted residual error
    std::vector<unsigned int> getPODBasisColumnsToAdd();

    /// Compute value of the functional on the coarse space
    double getCoarseFunctional();

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif
