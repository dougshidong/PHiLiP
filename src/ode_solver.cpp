
#include "ode_solver.h"
#include "linear_solver.h"


namespace PHiLiP
{
    template class ODESolver<PHILIP_DIM, double>;
    template class Explicit_ODESolver<PHILIP_DIM, double>;
    template class Implicit_ODESolver<PHILIP_DIM, double>;

    template <int dim, typename real>
    int ODESolver<dim, real>::steady_state ()
    {
        allocate_system ();

        dg->assemble_system ();
        residual_norm = dg->get_residual_l2norm();

        while (    residual_norm     > parameters->nonlinear_steady_residual_tolerance 
                && current_iteration < parameters->nonlinear_max_iterations )
        {
            ++current_iteration;

            if ( (current_iteration%parameters->print_iteration_modulo) == 0 )
            std::cout << " Iteration: " << current_iteration 
                      << " Residual norm: " << residual_norm
                      << std::endl;

            evaluate_solution_update ();
            solution += solution_update;

            dg->assemble_system ();
            residual_norm = dg->get_residual_l2norm();
        }
        return 1;
    }

    template <int dim, typename real>
    void Explicit_ODESolver<dim, real>::evaluate_solution_update ()
    {
        double dt = 1.0;
        //this->solution_update = dt*(this->dg->right_hand_side);
        this->solution_update = (this->dg->right_hand_side);
    }
    template <int dim, typename real>
    void Implicit_ODESolver<dim, real>::evaluate_solution_update ()
    {
        solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side, 
            this->solution_update);
    }

    template <int dim, typename real>
    void Explicit_ODESolver<dim, real>::allocate_system ()
    {
        unsigned int n_dofs = this->dg->dof_handler.n_dofs();
        this->solution_update.reinit(n_dofs);
        //solution.reinit(n_dofs);
        //right_hand_side.reinit(n_dofs);
    }
    template <int dim, typename real>
    void Implicit_ODESolver<dim, real>::allocate_system ()
    {
        solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side, 
            this->solution_update);
    }
}
