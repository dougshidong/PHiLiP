#ifndef __ASSEMBLE_ECSW_JACOBIAN__
#define __ASSEMBLE_ECSW_JACOBIAN__

#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include "dg/dg_base.hpp"
#include "pod_basis_base.h"
#include "parameters/all_parameters.h"
#include "assemble_ECSW_training_data_base.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Class for assembling NNLS problem (C matrix & d vector from the Jacobian of each snapshot) for finding
/// the weights for the ECSW hyper-reduction approach. NOTE: This class does not solve for the weights, but
/// A and b can be passed to the NNLS solver class.

template <int dim, int nstate>
class AssembleECSWJac: public AssembleECSWBase<dim,nstate>
{
public:
    /// Constructor
    AssembleECSWJac(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        MatrixXd snapshot_parameters_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type);

    /// Destructor
    ~AssembleECSWJac () {};

    /// Fill entries of A and b
    void build_problem() override;
};

} // HyperReduction namespace
} // PHiLiP namespace

#endif