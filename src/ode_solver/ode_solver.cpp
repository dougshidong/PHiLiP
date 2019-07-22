#include "ode_solver.h"
#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace ODE {

// can't just call ODESolver::steady_state...
// base class can't call virtual functions defined in the derived classes.
template <int dim, typename real>
int Implicit_ODESolver<dim,real>::steady_state ()
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;
    allocate_ode_system ();

    this->residual_norm = 1; // Always do at least 1 iteration
    double update_norm = 1; // Always do at least 1 iteration
    this->current_iteration = 0;

    while (    
               this->residual_norm     > ode_param.nonlinear_steady_residual_tolerance 
            && update_norm             > ode_param.nonlinear_steady_residual_tolerance 
            && this->current_iteration < ode_param.nonlinear_max_iterations )
    {
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " ********************************************************** "
                  << std::endl
                  << " Nonlinear iteration: " << this->current_iteration 
                  << " residual norm: " << this->residual_norm
                  << std::endl;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
        ODESolver<dim,real>::dg->assemble_residual_dRdW ();
        // (M/dt - dRdW) dw = R
        ODESolver<dim,real>::dg->system_matrix *= -1.0;

        const real time_step = 99.0;
        ODESolver<dim,real>::dg->add_mass_matrices(1.0/time_step);

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " Evaluating system update... " << std::endl;
        evaluate_solution_update ();


        ODESolver<dim,real>::dg->output_results_vtk(this->current_iteration);
        ODESolver<dim,real>::dg->solution += this->solution_update;

        update_norm = this->solution_update.l2_norm();
        this->residual_norm = ODESolver<dim,real>::dg->get_residual_l2norm();

        ++(this->current_iteration);

    }
    ODESolver<dim,real>::dg->output_results_vtk(this->current_iteration);
    return 1;
}
template <int dim, typename real>
int Explicit_ODESolver<dim,real>::steady_state ()
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;
    allocate_ode_system ();

    ODESolver<dim,real>::dg->assemble_residual_dRdW ();

    this->residual_norm = 1.0;
    this->current_iteration = 0;

    while (    this->residual_norm     > ode_param.nonlinear_steady_residual_tolerance 
            && this->current_iteration < ode_param.nonlinear_max_iterations )
    {
        ++this->current_iteration;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout 
                  << std::endl
                  << " Iteration: " << this->current_iteration 
                  << " Residual norm: " << this->residual_norm
                  << std::endl;

        evaluate_solution_update ();
        ODESolver<dim,real>::dg->solution += this->solution_update;

        ODESolver<dim,real>::dg->assemble_residual_dRdW ();
        this->residual_norm = ODESolver<dim,real>::dg->get_residual_l2norm();
    }
    return 1;
}

template <int dim, typename real>
void Explicit_ODESolver<dim,real>::evaluate_solution_update ()
{
    //this->solution_update = dt*(this->dg->right_hand_side);
    //this->solution_update = (this->dg->right_hand_side);
	double dt = 0.1;
    this->solution_update = 0;
    this->dg->global_mass_matrix.vmult(this->solution_update,this->dg->right_hand_side); //should be negative?
    this->solution_update *= dt;
}
template <int dim, typename real>
void Implicit_ODESolver<dim,real>::evaluate_solution_update ()
{
    solve_linear (
        this->dg->system_matrix,
        this->dg->right_hand_side, 
        this->solution_update,
        this->ODESolver<dim,real>::all_parameters->linear_solver_param);
}

template <int dim, typename real>
void Explicit_ODESolver<dim,real>::allocate_ode_system ()
{
    unsigned int n_dofs = this->dg->dof_handler.n_dofs();
    this->solution_update.reinit(n_dofs);
    this->dg->evaluate_inverse_mass_matrices();
    //solution.reinit(n_dofs);
    //right_hand_side.reinit(n_dofs);
}
template <int dim, typename real>
void Implicit_ODESolver<dim,real>::allocate_ode_system ()
{
    unsigned int n_dofs = this->dg->dof_handler.n_dofs();
    this->solution_update.reinit(n_dofs);
    this->dg->evaluate_mass_matrices();
}

//template <int dim, typename real>
//std::shared_ptr<ODESolver<dim,real>> ODESolverFactory<dim,real>::create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
//{
//    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
//    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<Explicit_ODESolver<dim,real>>();
//    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<Implicit_ODESolver<dim,real>>();
//    else {
//        std::cout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
//        return nullptr;
//    }
//}
template <int dim, typename real>
std::shared_ptr<ODESolver<dim,real>> ODESolverFactory<dim,real>::create_ODESolver(std::shared_ptr< DGBase<dim,real> > dg_input)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<Explicit_ODESolver<dim,real>>(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<Implicit_ODESolver<dim,real>>(dg_input);
    else {
        std::cout << "********************************************************************" << std::endl;
        std::cout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
        std::cout << "Solver type specified: " << ode_solver_type << std::endl;
        std::cout << "Solver type possible: " << std::endl;
        std::cout <<  ODEEnum::explicit_solver << std::endl;
        std::cout <<  ODEEnum::implicit_solver << std::endl;
        std::cout << "********************************************************************" << std::endl;
        return nullptr;
    }
}

template class ODESolver<PHILIP_DIM, double>;
template class Explicit_ODESolver<PHILIP_DIM, double>;
template class Implicit_ODESolver<PHILIP_DIM, double>;
template class ODESolverFactory<PHILIP_DIM, double>;

} // ODE namespace
} // PHiLiP namespace
