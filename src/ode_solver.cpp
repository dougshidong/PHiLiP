
#include "ode_solver.h"
#include "linear_solver.h"


namespace PHiLiP
{
    template class ODESolver<PHILIP_DIM, double>;
    template class Explicit_ODESolver<PHILIP_DIM, double>;
    template class Implicit_ODESolver<PHILIP_DIM, double>;
    template class ODESolverFactory<PHILIP_DIM, double>;


    // can't just call ODESolver::steady_state...
    // base class can't call virtual functions defined in the derived classes.
    template <int dim, typename real>
    int Implicit_ODESolver<dim,real>::steady_state ()
    {
        allocate_ode_system ();

        dg->assemble_system ();
        this->residual_norm = dg->get_residual_l2norm();
        this->current_iteration = 0;

        while (    this->residual_norm     > dg->parameters->nonlinear_steady_residual_tolerance 
                && this->current_iteration < dg->parameters->nonlinear_max_iterations )
        {
            ++this->current_iteration;

            if ( (this->current_iteration%dg->parameters->print_iteration_modulo) == 0 )
            std::cout << " Iteration: " << this->current_iteration 
                      << " Residual norm: " << this->residual_norm
                      << std::endl;

            evaluate_solution_update ();
            dg->solution += this->solution_update;

            dg->assemble_system ();
            this->residual_norm = dg->get_residual_l2norm();
        }
        return 1;
    }
    template <int dim, typename real>
    int Explicit_ODESolver<dim,real>::steady_state ()
    {
        allocate_ode_system ();

        dg->assemble_system ();
        this->residual_norm = dg->get_residual_l2norm();

        while (    this->residual_norm     > dg->parameters->nonlinear_steady_residual_tolerance 
                && this->current_iteration < dg->parameters->nonlinear_max_iterations )
        {
            ++this->current_iteration;

            if ( (this->current_iteration%dg->parameters->print_iteration_modulo) == 0 )
            std::cout << " Iteration: " << this->current_iteration 
                      << " Residual norm: " << this->residual_norm
                      << std::endl;

            evaluate_solution_update ();
            dg->solution += this->solution_update;

            dg->assemble_system ();
            this->residual_norm = dg->get_residual_l2norm();
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
    void Explicit_ODESolver<dim, real>::allocate_ode_system ()
    {
        unsigned int n_dofs = this->dg->dof_handler.n_dofs();
        this->solution_update.reinit(n_dofs);
        //solution.reinit(n_dofs);
        //right_hand_side.reinit(n_dofs);
    }
    template <int dim, typename real>
    void Implicit_ODESolver<dim, real>::allocate_ode_system ()
    {
        unsigned int n_dofs = this->dg->dof_handler.n_dofs();
        this->solution_update.reinit(n_dofs);
    }
    template <int dim, typename real>
    ODESolver<dim,real> *ODESolverFactory<dim,real>::create_ODESolver(Parameters::ODE::SolverType solver_type)
    {
        if(solver_type == Parameters::ODE::SolverType::explicit_solver) return new Explicit_ODESolver<dim, real>;
        if(solver_type == Parameters::ODE::SolverType::implicit_solver) return new Implicit_ODESolver<dim, real>;
        else {
            std::cout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
            return nullptr;
        }
    }
    template <int dim, typename real>
    ODESolver<dim,real> *ODESolverFactory<dim,real>::create_ODESolver(DiscontinuousGalerkin<dim,real> *dg_input)
    {
        if(dg_input->parameters->solver_type == Parameters::ODE::SolverType::explicit_solver) return new Explicit_ODESolver<dim, real>(dg_input);
        if(dg_input->parameters->solver_type == Parameters::ODE::SolverType::implicit_solver) return new Implicit_ODESolver<dim, real>(dg_input);
        else {
            std::cout << "********************************************************************" << std::endl;
            std::cout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
            std::cout << "Solver type specified: " << dg_input->parameters->solver_type << std::endl;
            std::cout << "Solver type possible: " << std::endl;
            std::cout <<  Parameters::ODE::SolverType::explicit_solver << std::endl;
            std::cout <<  Parameters::ODE::SolverType::implicit_solver << std::endl;
            std::cout << "********************************************************************" << std::endl;
            return nullptr;
        }
    }
}
