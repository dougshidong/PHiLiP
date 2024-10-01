#ifndef __ASSEMBLE_PROBLEM_ECSW__
#define __ASSEMBLE_PROBLEM_ECSW__

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

/// Class for assembling NNLS problem (C matrix & d vector from the residual of each snapshot) for finding
/// the weights for the ECSW hyper-reduction approach. NOTE: This class does not solve for the weights, but
/// A and b can be passed to the NNLS solver class.

/*
Reference for the ECSW training data assembly, find details in section 4.1 and Equation (17):
"Mesh sampling and weighting for the hyperreduction of nonlinear Petrov–Galerkin reduced-order models with local reduced-order bases"
Sebastian Grimberg, Charbel Farhat, Radek Tezaur, Charbel Bou-Mosleh
International Journal for Numerical Methods in Engineering, 2020
https://onlinelibrary.wiley.com/doi/10.1002/nme.6603
NOTE: The above presents the approach for training with the residual data
*/

template <int dim, int nstate>
class AssembleECSWRes: public AssembleECSWBase<dim,nstate>
{
public:
    /// Constructor
    AssembleECSWRes(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        MatrixXd snapshot_parameters_input,
        Parameters::ODESolverParam::ODESolverEnum ode_solver_type,
        Epetra_MpiComm &Comm);

    /// Destructor
    ~AssembleECSWRes () {};

    /// Fill entries of A and b
    void build_problem() override;
};

} // HyperReduction namespace
} // PHiLiP namespace

#endif