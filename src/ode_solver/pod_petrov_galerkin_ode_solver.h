#ifndef __POD_PETROV_GALERKIN_ODE_SOLVER__
#define __POD_PETROV_GALERKIN_ODE_SOLVER__

#include "dg/dg.h"
#include "reduced_order_ode_solver.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP {
namespace ODE {

/// POD-Petrov-Galerkin ODE solver derived from ReducedOrderODESolver.

/*
Reference for Petrov-Galerkin projection: Refer to Equation (23) in the following reference:
"Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
International Journal for Numerical Methods in Engineering, 2011
Petrov-Galerkin projection, test basis W = JV, pod basis V, system matrix J
W^T*J*V*p = -W^T*R
 */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODPetrovGalerkinODESolver: public ReducedOrderODESolver<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod); ///< Constructor.

    ///Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis) override;

    ///Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &epetra_system_matrix, Epetra_CrsMatrix &test_basis) override;
};

} // ODE namespace
} // PHiLiP namespace

#endif