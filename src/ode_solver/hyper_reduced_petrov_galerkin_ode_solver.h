#ifndef __HYPER_REDUCED_PETROV_GALERKIN_ODE_SOLVER__
#define __HYPER_REDUCED_PETROV_GALERKIN_ODE_SOLVER__

#include "dg/dg_base.hpp"
#include "ode_solver_base.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP {
namespace ODE {

/// Hyper-Reduced POD-Petrov-Galerkin ODE solver derived from ODESolver.

/*
Reference for Petrov-Galerkin projection: Refer to Equation (23) in the following reference:
"Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
International Journal for Numerical Methods in Engineering, 2011
Petrov-Galerkin projection, test basis W = JV, pod basis V, system matrix J
W^T*J*V*p = -W^T*R
 */

/*
Reference for the hyperreduction of the residual and the Jacobian via the ECSW approach: Refer to Equation (10) and (12) in:
"Mesh sampling and weighting for the hyperreduction of nonlinear Petrov–Galerkin reduced-order models with local reduced-order bases"
Sebastian Grimberg, Charbel Farhat, Radek Tezaur, Charbel Bou-Mosleh
International Journal for Numerical Methods in Engineering, 2020
https://onlinelibrary.wiley.com/doi/10.1002/nme.6603
Provides detail on how the hyperreduced residual and Jacobian are evaluated
*/

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class HyperReducedODESolver: public ODESolverBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    HyperReducedODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, Epetra_Vector weights);

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// ECSW hyper-reduction weights
    Epetra_Vector ECSW_weights;

    /// Destructor
    virtual ~HyperReducedODESolver() {};

    /// Evaluate steady state solution.
    int steady_state () override;

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
    /*Projection of initial conditions on reduced-order subspace, refer to Equation 19 in:
    Washabaugh, K. M., Zahr, M. J., & Farhat, C. (2016).
    On the use of discrete nonlinear reduced-order models for the prediction of steady-state flows past parametrically deformed complex geometries.
    In 54th AIAA Aerospace Sciences Meeting (p. 1814).
    */
    void allocate_ode_system () override;

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis);

    /// Generate hyper-reduced jacobian matrix
    std::shared_ptr<Epetra_CrsMatrix> generate_hyper_reduced_jacobian(const Epetra_CrsMatrix &system_matrix);

    /// Generate hyper-reduced residual
    std::shared_ptr<Epetra_Vector> generate_hyper_reduced_residual(Epetra_Vector epetra_right_hand_side, Epetra_CrsMatrix &test_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(Epetra_CrsMatrix &test_basis);

};

} // ODE namespace
} // PHiLiP namespace

#endif
