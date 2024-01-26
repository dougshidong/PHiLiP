#include "implicit_ode_solver.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
ImplicitODESolver<dim,real,MeshType>::ImplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        {}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    double step_length = 1.0;
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;

    // Assemble the Linear System
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw
    this->dg->system_matrix *= -1.0;

    if (pseudotime) {
        const double CFL = dt;
        this->dg->time_scaled_mass_matrices(CFL);
        this->dg->add_time_scaled_mass_matrices();
    } else {
        this->dg->add_mass_matrices(1.0/dt);
    }

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    // Solve the Linear System
    std::pair<unsigned int, double> linear_solver_iterations_and_residual;
    linear_solver_iterations_and_residual = solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side,
            this->solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    const auto old_solution = this->dg->solution;
    const double initial_residual = this->dg->get_residual_l2norm();

    if (this->ode_param.perform_linesearch == true) {
        step_length = linesearch();
        if (this->ode_param.perform_cfl_ramping == true) evaluate_cfl_new(step_length, initial_residual);
    } else {
        step_length = 0.;
        this->dg->solution.add(this->ode_param.relaxation_factor, this->solution_update);
        if (this->ode_param.perform_cfl_ramping == true) evaluate_cfl(step_length, initial_residual);
    }
    
    this->dg->solution = old_solution;
    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    const double new_residual = this->dg->get_residual_l2norm();

    this->update_norm = this->solution_update.l2_norm();

    std::clock_t cpu_timestamp = std::clock();

    this->pcout << " Track Convergence: " << 
	    std::setw(5) << this->current_iteration << 
	    std::setprecision(10) << std::fixed << "    " << cpu_timestamp << 
	    std::setprecision(10) << std::scientific << "    " << initial_residual << 
	    std::setprecision(10) << std::scientific << "    " << new_residual << 
	    std::setprecision(10) << std::scientific << "    " << this->update_norm << 
	    std::setprecision(10) << std::scientific << "    " << dt << 
	    std::setw(5) << "    " << linear_solver_iterations_and_residual.first << 
	    std::setprecision(10) << std::scientific << "    " << linear_solver_iterations_and_residual.second << 
	    std::endl;

    ++(this->current_iteration);

}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::evaluate_cfl_new (double step_length, double updated_residual)
{
    double alpha = 2.;
    double beta = 0.5;
    double minimum_step_length = 0.01;
    double minimum_ramp_step_length = 0.4;
    double CFL_maximum = pow(10.,5.);
    double CFL_reduction_factor = 0.5;

    // this->dg->assemble_residual ();
    // const double updated_residual = this->dg->get_residual_l2norm();
    const double CFL_minimum = pow(this->initial_nonlinear_residual_norm/updated_residual,beta);
    const double gamma = std::max((this->lastCFLupdated_nonlinear_residual_norm -updated_residual)/this->lastCFLupdated_nonlinear_residual_norm,0.);
    this->CFL_factor = pow(alpha,gamma);
    
    if (abs(step_length) > minimum_ramp_step_length) {
        this->CFL_current = std::min(this->CFL_current*this->CFL_factor,CFL_maximum);
        this->pcout << "1. Current CFL = " << this->CFL_current << std::endl;

    } else if (abs(step_length) < minimum_ramp_step_length && abs(step_length) > minimum_step_length) {
        this->CFL_current = std::max(this->CFL_current,CFL_minimum);
        this->pcout << "2. Current CFL = " << this->CFL_current << std::endl;

    } else {
        this->CFL_current = std::max(CFL_reduction_factor*this->CFL_current,CFL_minimum);
        this->pcout << "3. Current CFL = " << this->CFL_current << std::endl;

    }

    return;
}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::evaluate_cfl (double step_length, double initial_residual)
{
    double minimum_step_length = 0.09;

    this->dg->assemble_residual ();
    const double new_residual = this->dg->get_residual_l2norm();

    if (abs(step_length) > 0.) {
       if (abs(new_residual) < abs(initial_residual) 
           // && step_length == 1. 
           /* && abs(new_residual / initial_residual) <= 0.5 */ ) {
           this->CFL_factor *= 2.;
       } else if (abs(step_length) < minimum_step_length) {
           this->CFL_factor *= 0.1;
       }
    } else {

       double nonlinear_residual_ratio = initial_residual/new_residual;
       if (nonlinear_residual_ratio > 1.0) {
           this->CFL_factor = pow(nonlinear_residual_ratio,0.25);
       } else {
           this->CFL_factor = pow(nonlinear_residual_ratio,0.30);
       }
    }

    this->CFL_current = this->ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param.initial_time_step*this->CFL_factor;
    return;
}


template <int dim, typename real, typename MeshType>
double ImplicitODESolver<dim,real,MeshType>::linesearch_new ()
{
    const auto old_solution = this->dg->solution;
    double step_length = 1.0;

    const double step_reduction = 0.7;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    double minimum_step_length = 0.09;


    const double initial_residual = this->dg->get_residual_l2norm();

    // Compute unsteady residual norm
    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    auto mass_dw = this->solution_update;
    this->dg->time_scaled_global_mass_matrix.vmult(mass_dw, this->solution_update);
    this->dg->right_hand_side.add(step_length, mass_dw);    
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Start Line Search " << std::endl;
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline && step_length > minimum_step_length) {

        // Compute unsteady residual norm
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        this->dg->time_scaled_global_mass_matrix.vmult(mass_dw, this->solution_update);
        this->dg->right_hand_side.add(step_length, mass_dw);
        new_residual = this->dg->get_residual_l2norm();
        step_length *= step_reduction;

        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }

    if (step_length < minimum_step_length) {
        step_length = 0.;
    }
    this->pcout << " Final Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    return step_length;
}

template <int dim, typename real, typename MeshType>
double ImplicitODESolver<dim,real,MeshType>::linesearch ()
{
    const auto old_solution = this->dg->solution;
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    const double reduction_tolerance_2 = 2.0;

    const double initial_residual = this->dg->get_residual_l2norm();

    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Start Line Search " << std::endl;
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }
    // if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line Search (Case 2): Increase nonlinear residual tolerance by a factor " << std::endl;
        this->pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }
    /*
    if (iline == maxline) {
        this->CFL_factor *= 0.5;
        this->pcout << " Line Search (Case 3): Decrease CFL " << std::endl;
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        return 0.0;
    }*/

    if (iline == maxline) {
        this->pcout << " Line Search (Case 4): Reverse Search Direction " << std::endl;
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 5): Reverse Search Direction AND Increase nonlinear residual tolerance by a factor " << std::endl;
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
        //std::abort();
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 6): Decrease CFL " << std::endl;
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor : " << std::endl;
        this->dg->solution = old_solution;
     // this->CFL_factor *= 0.5;
    }

    return step_length;
}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);
}

template class ImplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ImplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ImplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
