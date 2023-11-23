#ifndef __ASSEMBLE_ECSW_JACOBIAN__
#define __ASSEMBLE_ECSW_JACOBIAN__

#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include "dg/dg.h"
#include "reduced_order/pod_basis_base.h"
#include "testing/pod_adaptive_sampling.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Class for assembling NNLS problem (C matrix & d vector from the residual of each snapshot) for finding
/// the weights for the ECSW hyper-reduction approach. NOTE: This class does not solve for the weights, but
/// A and b can be passed to the NNLS solver class.

template <int dim, int nstate>
class AssembleECSWJac
{
public:
    /// Constructor
    AssembleECSWJac(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        std::shared_ptr<Tests::AdaptiveSampling<dim,nstate>> parameter_sampling_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type);

    /// Destructor
    ~AssembleECSWJac () {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// dg
    std::shared_ptr<DGBase<dim,double>> dg;

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Sampling Class (currently being used for the reinitParams and snapshot_parameters)
    std::shared_ptr<Tests::AdaptiveSampling<dim,nstate>> parameter_sampling;

    const MPI_Comm mpi_communicator; ///< MPI communicator.

    /// ODE Solve Type/ Projection Type (galerkin or petrov-galerkins)
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type;

    /// Matrix for the NNLS Problem
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> A;

    /// RHS Vector for the NNLS Problem
    dealii::LinearAlgebra::ReadWriteVector<double> b;

    /// Generate Test Basis from the pod and snapshot info depending on the ode_solve_type (copied from the ODE solvers)
    std::shared_ptr<Epetra_CrsMatrix> local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis);
    
    /// Fill entries of A and b
    void build_problem();
};

}
}


#endif