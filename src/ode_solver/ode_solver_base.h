#ifndef __ODE_SOLVER_BASE__
#define __ODE_SOLVER_BASE__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/table_handler.h>
#include <iostream>
#include <deal.II/lac/vector.h>
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include <stdexcept>

namespace PHiLiP {
namespace ODE {

/// Base class ODE solver.

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

    /// Table used to output solution vector at each time step
    dealii::TableHandler solutions_table;

    /// Writes the ode solver steady state convergence data to a file
    void write_ode_solver_steady_state_convergence_data_to_table(
        const unsigned int current_iteration,
        const double current_residual,
        const std::shared_ptr <dealii::TableHandler> data_table) const;

    /// Evaluate steady state solution.
    virtual int steady_state ();

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

protected:
    /// CFL factor for (un)successful linesearches
    /** When the linesearch succeeds on its first try, double the CFL on top of
     *  the CFL ramping. If the linesearch fails and needs to look at the other direction
     *  or accept a higher residual, halve the CFL on top of the residual (de)ramping
     */
    double CFL_factor;

    double update_norm; ///< Norm of the solution update.
    double initial_residual_norm; ///< Initial residual norm.

    /// Solution update given by the ODE solver
    dealii::LinearAlgebra::distributed::Vector<double> solution_update;

public:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

protected:
    /// Input parameters.
    const Parameters::AllParameters *const all_parameters;

    /// Input ODE solver parameters
    const Parameters::ODESolverParam ode_param;

public:
    /// Useful for accurate time-stepping.
    /** This variable will change when step_in_time() is called. */
    double current_time;

    /// Current iteration.
    /** This variable will change when step_in_time() is called. */
    unsigned int current_iteration;

    /// Current desired time for output solution every dt time intervals
    /** This variable will change when advance_solution_time() is called
     *  if ODE parameter "output_solution_every_dt_time_intervals" > 0. */
    double current_desired_time_for_output_solution_every_dt_time_intervals;

    /// Getter for original_time_step
    double get_original_time_step() const;

    /// Getter for modified_time_step
    double get_modified_time_step() const;

protected:
    double original_time_step;///< Original time step before calling step_in_time
    double modified_time_step;///< Modified time step after calling step_in_time

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};
} // ODE namespace
} // PHiLiP namespace

#endif
