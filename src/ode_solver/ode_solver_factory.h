#ifndef __ODE_SOLVER_FACTORY__
#define __ODE_SOLVER_FACTORY__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis.h"
#include "ode_solver_base.h"

namespace PHiLiP {
namespace ODE {
/// Create specified ODE solver as ODESolverBase object
/** Factory design pattern whose job is to create the correct ODE solver
 */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ODESolverFactory
{
public:
    /// Creates either implicit or explicit ODE solver based on parameter value(no POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Creates either POD-Galerkin or POD-Petrov-Galerkin ODE solver based on parameter value (POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod);

    /// Creates either implicit or explicit ODE solver based on manual input (no POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Creates either POD-Galerkin or POD-Petrov-Galerkin ODE solver based on manual input (POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod);
};

} // ODE namespace
} // PHiLiP namespace

#endif
