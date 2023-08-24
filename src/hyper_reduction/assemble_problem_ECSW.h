#ifndef __ASSEMBLE_PROBLEM_ECSW__
#define __ASSEMBLE_PROBLEM_ECSW__

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

/// Class for assembling NNLS problem (C matrix & d vector from the residual) for ECSW
/* #if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif */
template <int dim, int nstate>
class AssembleECSW
{
public:
    /// Constructor
    AssembleECSW(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        std::shared_ptr<Tests::AdaptiveSampling<dim,nstate>> parameter_sampling_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type);

    /// Destructor
    ~AssembleECSW () {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// dg
    std::shared_ptr<DGBase<dim,double>> dg;

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Sampling Class
    std::shared_ptr<Tests::AdaptiveSampling<dim,nstate>> parameter_sampling;


    const MPI_Comm mpi_communicator; ///< MPI communicator.

    Parameters::ODESolverParam::ODESolverEnum ode_solver_type;

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> A;

    dealii::LinearAlgebra::ReadWriteVector<double> b;

    std::shared_ptr<Epetra_CrsMatrix> local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis);

    void build_problem();
};

}
}


#endif