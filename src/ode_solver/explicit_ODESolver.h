#ifndef __EXPLICIT_ODESOLVER__
#define __EXPLICIT_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

/// Explicit ODE solver derived from ODESolver.
class ExplicitODESolver: public ODESolverBase <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    ExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    virtual ~ExplicitODESolver() {}; ///< Destructor.

    /// Hard-coded way to play around with h-adaptivity.
    /// Not recommended to be used.
    int n_refine;

    /// Useful for accurate time-stepping.
    /** This variable will change when advance_solution_time() or step_in_time() is called. */
    double current_time;

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

protected:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// Input parameters.
    const Parameters::AllParameters *const all_parameters;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

} // ODE namespace
} // PHiLiP namespace

#endif
