#ifndef __ODESOLVER_H__
#define __ODESOLVER_H__

#include <deal.II/lac/vector.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "parameters/all_parameters.h"
#include "dg/dg.h"


namespace PHiLiP
{
    using namespace dealii;


    /// Base class ODE solver
    template <int dim, typename real>
    class ODESolver
    {
    public:
        ODESolver() {}; ///< Constructor
        ODESolver(int ode_solver_type); ///< Constructor
        /// Constructor
        ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input)
        :
        dg(dg_input),
        all_parameters(dg->all_parameters)
        {};
        virtual ~ODESolver() {}; ///< Destructor
        

        /// Virtual function to evaluate steady state solution
        virtual int steady_state () = 0;
        /// Virtual function to allocate the ODE system
        virtual void allocate_ode_system () = 0;

        int step_in_time(); ///< Step in time once.

        double residual_norm; ///< Current residual norm. Only makes sense for steady state

        unsigned int current_iteration; ///< Current iteration.

    protected:
        /// Virtual function to evaluate solution update
        virtual void evaluate_solution_update () = 0;

        /// Evaluate stable time-step
        /** Currently not used */
        void compute_time_step();

        /// Solution update given by the ODE solver
        Vector<real> solution_update;

        /// Solution vector.
        /** Currently not used. Might make it a vector pointing to dg->solution */
        Vector<real> solution;

        /// Right hand side vector.
        /** Currently not used. Might make it a vector pointing to dg->right_hand_side */
        Vector<real> right_hand_side;

        /// Smart pointer to DGBase
        std::shared_ptr<DGBase<dim,real>> dg;

        /// Pointer to AllParameters
        Parameters::AllParameters *all_parameters;

    }; // end of ODESolver class

    /// Implicit ODE solver
    /** Currently works to find steady state of linear problems.
     *  Need to add mass matrix to operator to handle nonlinear problems
     *  and time-accurate solutions.
     */
    template<int dim, typename real>
    class Implicit_ODESolver
        : public ODESolver<dim, real>
    {
    public:
        Implicit_ODESolver() {}; ///< Constructor.
        /// Constructor.
        Implicit_ODESolver(std::shared_ptr<DGBase<dim, real>> dg_input)
        :
        dg(dg_input),
        all_parameters(dg->all_parameters)
        {};
        ~Implicit_ODESolver() {}; ///< Destructor.
        void allocate_ode_system ();
        int steady_state ();
    protected:
        void evaluate_solution_update ();
        std::shared_ptr<DGBase<dim,real>> dg;

        Parameters::AllParameters *all_parameters;

    }; // end of Implicit_ODESolver class


    /// NON-WORKING Explicit ODE solver.
    /** Not tested. It worked a few commits ago before some major changes.
     *  Used to use assemble_implicit and just use the right-hand-side ignoring the system matrix
     */
    template<int dim, typename real>
    class Explicit_ODESolver
        : public ODESolver<dim, real>
    {
    public:
        Explicit_ODESolver() {};
        Explicit_ODESolver(std::shared_ptr<DGBase<dim, real>> dg_input)
        :
        dg(dg_input),
        all_parameters(dg->all_parameters)
        {};
        ~Explicit_ODESolver() {};
        void allocate_ode_system ();
        int steady_state ();
    protected:
        void evaluate_solution_update ();
        std::shared_ptr<DGBase<dim,real>> dg;

        Parameters::AllParameters *all_parameters;

    }; // end of Explicit_ODESolver class

    template <int dim, typename real>
    class ODESolverFactory
    {
    public:
        static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input);
        static std::shared_ptr<ODESolver<dim,real>> create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type);
    };


} // end of PHiLiP namespace

#endif

