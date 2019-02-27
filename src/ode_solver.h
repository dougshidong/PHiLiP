#ifndef __ODESOLVER_H__
#define __ODESOLVER_H__

#include <deal.II/lac/vector.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "parameters.h"
#include "dg.h"


namespace PHiLiP
{
    using namespace dealii;

    template <int dim, typename real>
    class ODESolver
    {
    public:
        ODESolver(int solver_type);
        virtual void allocate_system () = 0;
        int step_in_time();
        int steady_state ();

        double residual_norm;
        int    current_iteration;

    protected:
        virtual void evaluate_solution_update () = 0;
        void compute_time_step();

        Vector<real> solution;
        Vector<real> solution_update;
        Vector<real> right_hand_side;

        DiscontinuousGalerkin<dim,real> *dg;

        Parameters::AllParameters *parameters;

    }; // end of ODESolver class

    template<int dim, typename real>
    class Explicit_ODESolver
        : public ODESolver<dim, real>
    {
    public:
        void allocate_system ();
    protected:
        void evaluate_solution_update ();

    }; // end of Explicit_ODESolver class

    template<int dim, typename real>
    class Implicit_ODESolver
        : public ODESolver<dim, real>
    {
    public:
        void allocate_system ();
    protected:
        void evaluate_solution_update ();

    }; // end of Implicit_ODESolver class

} // end of PHiLiP namespace

#endif

