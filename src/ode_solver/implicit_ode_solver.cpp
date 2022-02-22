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
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw
    Parameters::ODESolverParam ode_param = ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param;

    this->dg->system_matrix *= -1.0;

    if (pseudotime) {
        const double CFL = dt;
        this->dg->time_scaled_mass_matrices(CFL);
        this->dg->add_time_scaled_mass_matrices();
    } else {
        this->dg->add_mass_matrices(1.0/dt);
    }

    if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side,
            this->solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    linesearch();

    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
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
    if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
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
    if (iline == maxline) {
        this->CFL_factor *= 0.5;
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        return 0.0;
    }

    if (iline == maxline) {
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
        this->pcout << " Line search failed. Trying to step in the opposite direction. " << std::endl;
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
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        this->CFL_factor *= 0.5;
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
