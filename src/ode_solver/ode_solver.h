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
    ODESolver(int ode_solver_type); ///< Constructor
    ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input); ///< Constructor
    virtual ~ODESolver() {}; ///< Destructor

    /// Useful for accurate time-stepping.
    /** This variable will change when advance_solution_time() or step_in_time() is called. */
    double current_time;

    /// Evaluate steady state solution
    int steady_state ();

    void initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree);


    /// Virtual function to advance solution to time+dt
    int advance_solution_time (double time_advance);

    /// Virtual function to allocate the ODE system
    virtual void allocate_ode_system () = 0;

    double residual_norm; ///< Current residual norm. Only makes sense for steady state
    double residual_norm_decrease; ///< Current residual norm normalized by initial residual. Only makes sense for steady state

    unsigned int current_iteration; ///< Current iteration.

protected:
    double update_norm;
    double initial_residual_norm;

    /// Virtual function to evaluate solution update
    virtual void step_in_time(real dt) = 0;

    /// Evaluate stable time-step
    /** Currently not used */
    void compute_time_step();

    /// Solution update given by the ODE solver
    dealii::LinearAlgebra::distributed::Vector<double> solution_update;

    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;

    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real>> dg;

    const Parameters::AllParameters *const all_parameters;

    const MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;


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
    void allocate_ode_system ();
protected:
    void step_in_time(real dt);
    using ODESolver<dim,real>::pcout;

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
    Explicit_ODESolver() = delete;
    Explicit_ODESolver(std::shared_ptr<DGBase<dim, real>> dg_input)
    : ODESolver<dim,real>::ODESolver(dg_input) 
    {};
    ~Explicit_ODESolver() {};
    void allocate_ode_system ();

protected:
    void step_in_time(real dt);
    using ODESolver<dim,real>::pcout;
}; // end of Explicit_ODESolver class

/// Creates and assemble Explicit_ODESolver or Implicit_ODESolver as ODESolver based on input.
template <int dim, typename real>
class ODESolverFactory
{
public:
    static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input);
    static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type);
};


} // ODE namespace
} // PHiLiP namespace

#endif

