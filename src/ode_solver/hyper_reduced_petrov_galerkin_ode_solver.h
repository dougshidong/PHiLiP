#ifndef __HYPER_REDUCED_PETROV_GALERKIN_ODE_SOLVER__
#define __HYPER_REDUCED_PETROV_GALERKIN_ODE_SOLVER__

#include "dg/dg_base.hpp"
#include "ode_solver_base.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP {
namespace ODE {

/// Hyper-Reduced POD-Petrov-Galerkin ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class HyperReducedODESolver: public ODESolverBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    HyperReducedODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, Epetra_Vector weights); ///< Constructor.

    ///POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// ECSW hyper-reduction weights
    Epetra_Vector xi;

    /// Destructor
    virtual ~HyperReducedODESolver() {};

    /// Evaluate steady state solution.
    int steady_state () override;

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
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
