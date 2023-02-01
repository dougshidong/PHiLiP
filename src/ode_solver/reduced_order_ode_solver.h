#ifndef __REDUCED_ORDER_ODE_SOLVER__
#define __REDUCED_ORDER_ODE_SOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP {
namespace ODE {

/// POD-Petrov-Galerkin ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ReducedOrderODESolver: public ODESolverBase<dim, real, MeshType>
{
protected:
    /// Default constructor that will set the constants.
    ReducedOrderODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod); ///< Constructor.

public:
    ///POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Destructor
    virtual ~ReducedOrderODESolver() {};

    /// Evaluate steady state solution.
    int steady_state () override;

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
    void allocate_ode_system () override;

    /// Generate test basis depending on which projection is used
    virtual std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis) = 0;

    /// Generate the reduced left-hand side depending on which projection is used
    virtual std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &epetra_system_matrix, Epetra_CrsMatrix &test_basis) = 0;

};

} // ODE namespace
} // PHiLiP namespace

#endif
