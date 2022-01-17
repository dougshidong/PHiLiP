#ifndef __POD_ADAPTATION__
#define __POD_ADAPTATION__

#include <fstream>
#include <iostream>
#include <filesystem>

#include <typeinfo>

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

    /// Smart pointer to POD
    std::shared_ptr<ProperOrthogonalDecomposition::CoarsePOD> coarsePOD;

    std::shared_ptr<ProperOrthogonalDecomposition::SpecificPOD> finePOD;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

    /// Linear solver parameters.
    Parameters::LinearSolverParam linear_solver_param;

    DealiiVector dualWeightedResidual;

    double error;

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

public:
    /// Constructor
    PODAdaptation(std::shared_ptr<DGBase<dim,double>> &_dg, Functional<dim,nstate,double> &_functional, std::shared_ptr<ProperOrthogonalDecomposition::CoarsePOD> _coarsePOD, std::shared_ptr<ProperOrthogonalDecomposition::SpecificPOD> _finePOD);

    /// Constructor not specifying number of basis functions
    PODAdaptation();

    /// Destructor
    ~PODAdaptation () {};

    void getReducedGradient(DealiiVector &reducedGradient);

    void applyReducedJacobianTranspose(DealiiVector &reducedAdjoint, DealiiVector &reducedGradient);

    void simplePODAdaptation();

    void progressivePODAdaptation();

    void getDualWeightedResidual();

    std::vector<unsigned int> getPODBasisColumnsToAdd();

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};

}
}

#endif
