#ifndef __ODE_SOLVER_BASE__
#define __ODE_SOLVER_BASE__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/table_handler.h>
#include <iostream>

#include <deal.II/lac/vector.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"



namespace PHiLiP {
namespace ODE {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ODESolverBase
{
public:
    /// Default constructor that will set the constants.
    ODESolverBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    virtual ~ODESolverBase() {}; ///< Destructor.

    /// Hard-coded way to play around with h-adaptivity.
    /// Not recommended to be used.
    int n_refine;

    /// Useful for accurate time-stepping.
    /** This variable will change when advance_solution_time() or step_in_time() is called. */
    double current_time;

    /// Table used to output solution vector at each time step
    dealii::TableHandler solutions_table;

    /// Evaluate steady state solution.
    int steady_state ();

    /// Ramps up the solution from p0 all the way up to the given global_final_poly_degree.
    /** This first interpolates the current solution to the P0 space as an initial solution.
     *  The P0 is then solved, interpolated to P1, and the process is repeated until the
     *  desired polynomial is solved.
     *
     *  This is mainly usely for grid studies.
     */
    void initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree);


    /// Checks whether the DG vector has valid values.
    /** By default, the DG solution vector is initialized with the lowest possible value.
     */
    void valid_initial_conditions () const;

    /// Function to advance solution to time+dt
    int advance_solution_time (double time_advance);

    /// Virtual function to evaluate solution update
    virtual void step_in_time(real dt, const bool pseudotime) = 0;

    /// Virtual function to allocate the ODE system
    virtual void allocate_ode_system () = 0;

    double residual_norm; ///< Current residual norm. Only makes sense for steady state
    double residual_norm_decrease; ///< Current residual norm normalized by initial residual. Only makes sense for steady state

    unsigned int current_iteration; ///< Current iteration.

protected:
    /// Hard-coded way to play around with h-adaptivity.
    /// Not recommended to be used.
    bool refine;

    /// CFL factor for (un)successful linesearches
    /** When the linesearch succeeds on its first try, double the CFL on top of
     *  the CFL ramping. If the linesearch fails and needs to look at the other direction
     *  or accept a higher residual, halve the CFL on top of the residual (de)ramping
     */
    double CFL_factor;

    double update_norm; ///< Norm of the solution update.
    double initial_residual_norm; ///< Initial residual norm.

    /// Evaluate stable time-step
    /** Currently not used */
    void compute_time_step();

    /// Solution update given by the ODE solver
    dealii::LinearAlgebra::distributed::Vector<double> solution_update;

    /// Stores the various RK stages.
    /** Currently hard-coded to RK4.
     */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;

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
