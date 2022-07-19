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
    {
        const bool compute_dRdW = true;
        //this->dg->assemble_residual(compute_dRdW);
        const bool compute_dRdX=false;
        const bool compute_d2R=false;
        const double CFL_mass = 0.0;
        const bool compute_p0_dRdW=false;//true;
        this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, CFL_mass, compute_p0_dRdW);
    }
    {
        const bool compute_dRdW =false;
        const bool compute_dRdX=false;
        const bool compute_d2R=false;
        const double CFL_mass = 0.0;
        const bool compute_p0_dRdW=false;
        this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, CFL_mass, compute_p0_dRdW);
    }
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw

    this->dg->system_matrix *= -1.0;

    auto dRdWinv_mass_R = this->solution_update;
    if (pseudotime) {
        const double CFL = dt;
        this->dg->time_scaled_mass_matrices(CFL);
        this->dg->add_time_scaled_mass_matrices();

        // Pseudo-transient continuation (PTC) residual smoothing procedure
        // Mavriplis, D. J. “A Residual Smoothing Strategy for Accelerating Newton Method Continuation.”
        // Computers & Fluids, 2021-02, p. 104859.
        // https://doi.org/10.1016/j.compfluid.2021.104859.
        const bool PTC_residual_smoothing = false;
        if (PTC_residual_smoothing) {
            solve_linear (
                    this->dg->system_matrix,
                    this->dg->right_hand_side,
                    this->solution_update,
                    this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

            this->dg->time_scaled_global_mass_matrix.vmult(dRdWinv_mass_R, this->solution_update);
            this->dg->right_hand_side.add(1.0,dRdWinv_mass_R);
        }

    } else {
        this->dg->add_mass_matrices(1.0/dt);
    }

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
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

    const double initial_l2residual = this->dg->get_residual_l2norm();

    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();

    auto mass_dw = this->solution_update;
    this->dg->time_scaled_global_mass_matrix.vmult(mass_dw, this->solution_update);
    this->dg->right_hand_side.add(step_length, mass_dw);

    double new_l2residual = this->dg->get_residual_l2norm();
    this->pcout << " Step length " << step_length << ". Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_l2residual > initial_l2residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();

        this->dg->right_hand_side.add(step_length, mass_dw);

        new_l2residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line search failed. Will accept any valid l2residual less than " << reduction_tolerance_2 << " times the current " << initial_l2residual << "l2residual. " << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_l2residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
        for (iline = 0; iline < maxline && new_l2residual > initial_l2residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            this->dg->right_hand_side.add(step_length, mass_dw);
            new_l2residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
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
        this->dg->right_hand_side.add(step_length, mass_dw);
        new_l2residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
        for (iline = 0; iline < maxline && new_l2residual > initial_l2residual * reduction_tolerance_1 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            this->dg->right_hand_side.add(step_length, mass_dw);
            new_l2residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line search failed. Trying to step in the opposite direction. " << std::endl;
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        this->dg->right_hand_side.add(step_length, mass_dw);
        new_l2residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
        for (iline = 0; iline < maxline && new_l2residual > initial_l2residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            this->dg->right_hand_side.add(step_length, mass_dw);
            new_l2residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old l2residual: " << initial_l2residual << " New l2residual: " << new_l2residual << std::endl;
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
