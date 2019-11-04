#ifndef __ODESOLVER_H__
#define __ODESOLVER_H__

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>

#include "parameters/all_parameters.h"
#include "dg/dg.h"


namespace PHiLiP {
namespace ODE {

/// Base class ODE solver.
/** The ODE solver assumes the following form of the equations are given
 *  \f[
 *      \frac{\partial \mathbf{u}}{\partial t} = \mathbf{R}(\mathbf{u})
 *  \f]
 */
template <int dim, typename real>
class ODESolver
{
public:
    ODESolver(int ode_solver_type); ///< Constructor.
    ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input); ///< Constructor.
    virtual ~ODESolver() {}; ///< Destructor.

    /// Useful for accurate time-stepping.
    /** This variable will change when advance_solution_time() or step_in_time() is called. */
    double current_time;

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


    /// Virtual function to advance solution to time+dt
    int advance_solution_time (double time_advance);

    /// Virtual function to allocate the ODE system
    virtual void allocate_ode_system () = 0;

    double residual_norm; ///< Current residual norm. Only makes sense for steady state
    double residual_norm_decrease; ///< Current residual norm normalized by initial residual. Only makes sense for steady state

    unsigned int current_iteration; ///< Current iteration.

protected:
    double update_norm; ///< Norm of the solution update.
    double initial_residual_norm; ///< Initial residual norm.

    /// Virtual function to evaluate solution update
    virtual void step_in_time(real dt) = 0;

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
    std::shared_ptr<DGBase<dim,real>> dg;

    /// Input parameters.
    const Parameters::AllParameters *const all_parameters;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0


}; // end of ODESolver class

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
template<int dim, typename real>
class Implicit_ODESolver
    : public ODESolver<dim, real>
{
public:
    Implicit_ODESolver() = delete; ///< Constructor.
    /// Constructor.
    Implicit_ODESolver(std::shared_ptr<DGBase<dim, real>> dg_input)
    :
    ODESolver<dim,real>::ODESolver(dg_input)
    {};
    ~Implicit_ODESolver() {}; ///< Destructor.
    /// Allocates ODE system based on given DGBase.
    /** Basically allocates solution vector and asks DGBase to evaluate the mass matrix.
     */
    void allocate_ode_system ();
protected:
    /// Advances the solution in time by \p dt.
    void step_in_time(real dt);

    using ODESolver<dim,real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // end of Implicit_ODESolver class


/// NON-TESTED Explicit ODE solver derived from ODESolver.
/** Not tested. It worked a few commits ago before some major changes.
 *  Used to use assemble_implicit and just use the right-hand-side ignoring the system matrix
 */
template<int dim, typename real>
class Explicit_ODESolver
    : public ODESolver<dim, real>
{
public:
    /// Constructor.
    Explicit_ODESolver(std::shared_ptr<DGBase<dim, real>> dg_input)
    : ODESolver<dim,real>::ODESolver(dg_input) 
    {};
    /// Destructor.
    ~Explicit_ODESolver() {};
    /// Allocates ODE system based on given DGBase.
    /** Basically allocates rk-stages and solution vector and asks DGBase to evaluate the inverse mass matrix.
     */
    void allocate_ode_system ();

protected:
    void step_in_time(real dt); ///< Advances the solution in time by \p dt.
    using ODESolver<dim,real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
}; // end of Explicit_ODESolver class

/// Creates and assemble Explicit_ODESolver or Implicit_ODESolver as ODESolver based on input.
template <int dim, typename real>
class ODESolverFactory
{
public:
    /// Creates the ODE solver given a DGBase.
    /** The input parameters are copied from the DGBase since they should be consistent
     */
    static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input);
    // static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type);
};


} // ODE namespace
} // PHiLiP namespace

#endif

