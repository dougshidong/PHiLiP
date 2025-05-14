#ifndef __ASSEMBLE_ECSW_BASE__
#define __ASSEMBLE_ECSW_BASE__

#include <eigen/Eigen/Dense>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include "dg/dg_base.hpp"
#include "pod_basis_base.h"
#include "parameters/all_parameters.h"
#include "multi_core_helper_functions.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Base class for assembling the ECSW training data. Training data can be residual-based or Jacobian-based.
/// NOTE: This class does not solve for the weights, but A and b can be passed to the NNLS solver class.

/*
Reference for the ECSW training data assembly, find details in section 4.1 and Equation (17):
"Mesh sampling and weighting for the hyperreduction of nonlinear Petrov–Galerkin reduced-order models with local reduced-order bases"
Sebastian Grimberg, Charbel Farhat, Radek Tezaur, Charbel Bou-Mosleh
International Journal for Numerical Methods in Engineering, 2020
https://onlinelibrary.wiley.com/doi/10.1002/nme.6603
NOTE: The above presents the approach for training with the residual data
*/

template <int dim, int nstate>
class AssembleECSWBase
{
public:
    /// Constructor
    AssembleECSWBase(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        MatrixXd snapshot_parameters_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type,
        Epetra_MpiComm &Comm);

    /// Destructor
    virtual ~AssembleECSWBase () {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// dg
    std::shared_ptr<DGBase<dim,double>> dg;

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

    /// ODE Solve Type/ Projection Type (galerkin or petrov-galerkin)
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type;

    /// Epetra Communicator Object with MPI
    Epetra_MpiComm Comm_;

    /// Matrix for the NNLS Problem
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> A_T;

    /// RHS Vector for the NNLS Problem
    dealii::LinearAlgebra::ReadWriteVector<double> b;

    /// Vector of parameter-ROMTestLocation pairs
    mutable std::vector<dealii::LinearAlgebra::distributed::Vector<double>> fom_locations;

    /// Generate Test Basis from the pod and snapshot info depending on the ode_solve_type (copied from the ODE solvers)
    std::shared_ptr<Epetra_CrsMatrix> local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(const RowVectorXd& parameter) const;

    /// Update POD and Snapshot Parameters
    void update_snapshots(dealii::LinearAlgebra::distributed::Vector<double> fom_solution);

    /// Update POD and Snapshot Parameters
    void update_POD_snaps(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_update, MatrixXd snapshot_parameters_update);

    /// Fill entries of A and b
    virtual void build_problem() = 0;
};

} // HyperReduction namespace
} // PHiLiP namespace

#endif