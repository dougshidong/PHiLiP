#ifndef __IMPLICIT_ODESOLVER__
#define __IMPLICIT_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace ODE {

/// Implicit ODE solver derived from ODESolver.
/** Currently works to find steady state of linear problems.
 *  Need to add mass matrix to operator to handle nonlinear problems
 *  and time-accurate solutions.
 *
 *  Uses backward-Euler by linearizing the residual
 *  \f[
 *      \mathbf{R}(\mathbf{u}^{n+1}) = \mathbf{R}(\mathbf{u}^{n}) +
 *      \left. \frac{\partial \mathbf{R}}{\partial \mathbf{u}} \right|_{\mathbf{u}^{n}} (\mathbf{u}^{n+1} - \mathbf{u}^{n})
 *  \f]
 *  \f[
 *      \frac{\partial \mathbf{u}}{\partial t} = \mathbf{R}(\mathbf{u}^{n+1})
 *  \f]
 *  \f[
 *      \frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t} = \mathbf{R}(\mathbf{u}^{n}) +
 *      \left. \frac{\partial \mathbf{R}}{\partial \mathbf{u}} \right|_{\mathbf{u}^{n}} (\mathbf{u}^{n+1} - \mathbf{u}^{n})
 *  \f]
 */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ImplicitODESolver: public ODESolverBase <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    ImplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    /// Destructor.
    ~ImplicitODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

    /// Line search algorithm
    double linesearch ();

};

} // ODE namespace
} // PHiLiP namespace

#endif
