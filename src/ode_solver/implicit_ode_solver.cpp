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

    solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side,
            this->solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);
/*
    const double threshold_pressure_update_fraction = 0.1;
    const double step_length_min = 0.001;

    const double step_length = linesearch_pressure_based(this->dg->solution, this->solution_update, threshold_pressure_update_fraction);

    if(step_length > step_length_min)
    {
        this->dg->solution.add(step_length, this->solution_update);
    }
    else
    {
        this->dg->update_solution_with_min_steplength_elsewhere(this->dg->solution, this->solution_update, step_length, step_length_min, threshold_pressure_update_fraction);
    }
    this->dg->solution.update_ghost_values();
*/
    linesearch();
    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
}

template <int dim, typename real, typename MeshType>
double ImplicitODESolver<dim,real,MeshType>::linesearch_pressure_based(
    const dealii::LinearAlgebra::distributed::Vector<double> &solution, 
    const dealii::LinearAlgebra::distributed::Vector<double> &solution_update,
    const double threshold_pressure_update_fraction)
{
    double step_length = 0.5;
    const double step_update_factor = 2.0;
    const double time_step_update_factor = 2.0;

    const int max_linesearches = 10;
    int isearch = 0;
    for(isearch = 0; isearch < max_linesearches; ++isearch)
    {
        dealii::LinearAlgebra::distributed::Vector<double> solution_new = solution;
        solution_new.add(step_length, solution_update);
        const bool pressure_update_is_below_threshold = 
                this->dg->is_pressure_update_below_threshold(solution, solution_new, threshold_pressure_update_fraction);
        
        this->pcout<<" Step length "<<step_length<<" . Pressure update is below threshold = "<<pressure_update_is_below_threshold<<std::endl;

        if(pressure_update_is_below_threshold) {break;}
        step_length /=step_update_factor; 
    }

    if(isearch == 0) {++n_linesearches_with_iline0;}
    if(n_linesearches_with_iline0 == 10)
    {
        n_linesearches_with_iline0 = 0;
        this->CFL_factor*=time_step_update_factor;
        this->CFL_factor = std::min(this->CFL_factor, 1.0e10); 
    }

    if(isearch == max_linesearches)
    //if(isearch!=0)
    {
        this->CFL_factor /= time_step_update_factor;
        this->pcout << " Updated dt to "<<this->CFL_factor << std::endl;
    }

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
