#ifndef __ODE_SOLVER_FACTORY__
#define __ODE_SOLVER_FACTORY__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "reduced_order/pod_basis_base.h"
#include "ode_solver_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"

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
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod);

    /// Creates either implicit or explicit ODE solver based on manual input (no POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Creates either POD-Galerkin or POD-Petrov-Galerkin ODE solver based on manual input (POD basis given)
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod);

    /// Output error message for Implicit and Explicit solver
    static void display_error_ode_solver_factory(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, bool reduced_order);
    
    /// Creates an ODESolver object based on the specified RK method, including derived classes
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Creates an RKTableau object based on the specified RK method
    static std::shared_ptr<RKTableauBase<dim,real,MeshType>> create_RKTableau(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input);

};

} // ODE namespace
} // PHiLiP namespace

#endif
