#ifndef __ODE_SOLVER_FACTORY__
#define __ODE_SOLVER_FACTORY__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "pod/proper_orthogonal_decomposition.h"
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
    /// Creates the ODE solver given a DGBase.
    static std::shared_ptr<ODESolverBase<dim,real,MeshType>> create_ODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD> pod = NULL);
};

} // ODE namespace
} // PHiLiP namespace

#endif
