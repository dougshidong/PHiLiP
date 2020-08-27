#include <deal.II/distributed/solution_transfer.h>

#include "ode_solver.h"

#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real>
ODESolver<dim,real>::ODESolver(std::shared_ptr< DGBase<dim, real> > dg_input)
    : current_time(0.0)
    , dg(dg_input)
    , all_parameters(dg->all_parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}

template <int dim, typename real>
void ODESolver<dim,real>::initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree)
{
    pcout << " ************************************************************************ " << std::endl;
    pcout << " Initializing DG with global polynomial degree = " << global_final_poly_degree << " by ramping from degree 0 ... " << std::endl;
    pcout << " ************************************************************************ " << std::endl;

    for (unsigned int degree = 0; degree <= global_final_poly_degree; degree++) {
        pcout << " ************************************************************************ " << std::endl;
        pcout << " Ramping degree " << degree << " until p=" << global_final_poly_degree << std::endl;
        pcout << " ************************************************************************ " << std::endl;

        dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
        old_solution.update_ghost_values();

        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(old_solution);

        dg->set_all_cells_fe_degree(degree);
        //dg->triangulation->execute_coarsening_and_refinement();
        // Required even if no mesh refinement takes place
        //dg->triangulation->execute_coarsening_and_refinement();
        //dg->triangulation->refine_global (1);
        dg->allocate_system ();

        //old_solution.print(pcout.get_stream());
        dg->solution.zero_out_ghosts();
        solution_transfer.interpolate(dg->solution);
        dg->solution.update_ghost_values();
        //dg->solution.print(pcout.get_stream());

        //dealii::LinearAlgebra::distributed::Vector<double> new_solution(dg->locally_owned_dofs, MPI_COMM_WORLD);
        //new_solution.zero_out_ghosts();
        //solution_transfer.interpolate(new_solution);
        //new_solution.update_ghost_values();
        //new_solution.print(pcout.get_stream());

        steady_state();
    }
}

template <int dim, typename real>
int ODESolver<dim,real>::steady_state ()
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;
    pcout << " Performing steady state analysis... " << std::endl;
    allocate_ode_system ();

    this->residual_norm_decrease = 1; // Always do at least 1 iteration
    update_norm = 1; // Always do at least 1 iteration
    this->current_iteration = 0;
    if (ode_param.output_solution_every_x_steps >= 0) this->dg->output_results_vtk(this->current_iteration);

    pcout << " Evaluating right-hand side and setting system_matrix to Jacobian before starting iterations... " << std::endl;
    this->dg->assemble_residual ();
    initial_residual_norm = this->dg->get_residual_l2norm();
    this->residual_norm = initial_residual_norm;
    pcout << " ********************************************************** "
          << std::endl
          << " Initial absolute residual norm: " << this->residual_norm
          << std::endl;

    CFL = all_parameters->ode_solver_param.initial_time_step;

    // Output initial solution
    while (    this->residual_norm     > ode_param.nonlinear_steady_residual_tolerance 
            && this->residual_norm_decrease > ode_param.nonlinear_steady_residual_tolerance 
            //&& update_norm             > ode_param.nonlinear_steady_residual_tolerance 
            && this->current_iteration < ode_param.nonlinear_max_iterations )
    {
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose
            && (this->current_iteration%ode_param.print_iteration_modulo) == 0 
            && dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 )
        {
            pcout.set_condition(true);
        } else {
            pcout.set_condition(false);
        }
        pcout << " ********************************************************** "
                  << std::endl
                  << " Nonlinear iteration: " << this->current_iteration
                  << " Normalized residual norm: " << this->residual_norm / this->initial_residual_norm
                  << std::endl;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
            pcout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
        }

        double dt = std::max(CFL,all_parameters->ode_solver_param.initial_time_step);
        if (this->residual_norm_decrease < 1.0) {
            dt *= pow((1.0-std::log10(this->residual_norm_decrease)*ode_param.time_step_factor_residual), ode_param.time_step_factor_residual_exp);
        }
        dt = std::max(dt,CFL);
        pcout << "CFL = " << CFL << " Time step = " << dt << std::endl;

        step_in_time(dt);

        this->dg->assemble_residual ();
        this->residual_norm = this->dg->get_residual_l2norm();
        this->residual_norm_decrease = this->residual_norm / this->initial_residual_norm;

        ++(this->current_iteration);

        if (ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }
    }

    pcout << " ********************************************************** "
          << std::endl
          << " ODESolver steady_state stopped at"
          << std::endl
          << " Nonlinear iteration: " << this->current_iteration
          << " residual norm: " << this->residual_norm
          << std::endl
          << " ********************************************************** "
          << std::endl;

    return 1;
}

template <int dim, typename real>
int ODESolver<dim,real>::advance_solution_time (double time_advance)
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;

    const unsigned int number_of_time_steps = static_cast<int>(ceil(time_advance/ode_param.initial_time_step));
    const double constant_time_step = time_advance/number_of_time_steps;

    pcout
        << " Advancing solution by " << time_advance << " time units, using "
        << number_of_time_steps << " iterations of size dt=" << constant_time_step << " ... " << std::endl;
    allocate_ode_system ();

    this->current_iteration = 0;

    // Output initial solution
    this->dg->output_results_vtk(this->current_iteration);

    while (this->current_iteration < number_of_time_steps)
    {
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        pcout << " ********************************************************** "
              << std::endl
              << " Iteration: " << this->current_iteration + 1
              << " out of: " << number_of_time_steps
              << std::endl;
    }
        dg->assemble_residual(false);

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        pcout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
    }

        step_in_time(constant_time_step);


    if (this->current_iteration%ode_param.print_iteration_modulo == 0) {
        this->dg->output_results_vtk(this->current_iteration);
    }
        ++(this->current_iteration);


        //this->dg->output_results_vtk(this->current_iteration);
    }
    return 1;
}

template <int dim, typename real>
void Implicit_ODESolver<dim,real>::step_in_time (real dt)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;

    this->dg->system_matrix *= -1.0;

    //this->dg->add_mass_matrices(1.0/dt);
    const double dt_scale = dt;
    this->dg->time_scaled_mass_matrices(dt_scale);
    this->dg->add_time_scaled_mass_matrices();

    if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        pcout << " Evaluating system update... " << std::endl;
    }

    solve_linear (
        this->dg->system_matrix,
        this->dg->right_hand_side, 
        this->solution_update,
        this->ODESolver<dim,real>::all_parameters->linear_solver_param);

    //this->dg->solution += this->solution_update;
    linesearch();

    this->update_norm = this->solution_update.l2_norm();
}

template <int dim, typename real>
double Implicit_ODESolver<dim,real>::linesearch ()
{
    const auto old_solution = this->dg->solution;
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 30;
    const double reduction_tolerance = 1.5;

    const double initial_residual = this->dg->get_residual_l2norm();

    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    double new_residual = this->dg->get_residual_l2norm();

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual; ++iline) {
        pcout << " Step length " << step_length << " did not reduce residual. Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
    }
    if (step_length > std::pow(step_reduction,maxline/2)) {
        //this->CFL *= 1.2;
    } else {
        //this->CFL *= 0.5;
    }
    if (iline == maxline) {
        step_length = 1.0;
        pcout << " Step length " << step_length << " accepting any give valid residual. Old residual: " << initial_residual << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance ; ++iline) {
            pcout << " Step length " << step_length << " did not give valid residual. Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
        }
        //this->CFL *= 0.5;
    }

    if (iline == maxline) {
        pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        step_length = -1.0;
        pcout << " Step length " << step_length << " accepting any give valid residual. Old residual: " << initial_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual; ++iline) {
            pcout << " Step length " << step_length << " did not reduce residual. Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
        }
        if (iline == maxline) {
            step_length = 1.0;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance ; ++iline) {
                pcout << " Step length " << step_length << " did not give valid residual. Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
                step_length = step_length * step_reduction;
                this->dg->solution = old_solution;
                this->dg->solution.add(step_length, this->solution_update);
                this->dg->assemble_residual ();
                new_residual = this->dg->get_residual_l2norm();
            }
        }
        //this->CFL *= 0.5;
        //std::abort();
    }

    return step_length;
}

template <int dim, typename real>
void Explicit_ODESolver<dim,real>::step_in_time (real dt)
{
    // this->dg->assemble_residual (); // Not needed since it is called in the base class for time step
    this->current_time += dt;
    const int rk_order = 1;
    if (rk_order == 1) {
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->update_norm = this->solution_update.l2_norm();
        this->dg->solution.add(dt,this->solution_update);
    } else if (rk_order == 3) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;

        // Stage 1
        pcout<< "Stage 1... " << std::flush;
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[1] = this->rk_stage[0];
        this->rk_stage[1].add(dt,this->solution_update);

        this->dg->solution = this->rk_stage[1];

        // Stage 2
        pcout<< "2... " << std::flush;
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[2] = this->rk_stage[0];
        this->rk_stage[2] *= 0.75;
        this->rk_stage[2].add(0.25, this->rk_stage[1]);
        this->rk_stage[2].add(0.25*dt, this->solution_update);

        this->dg->solution = this->rk_stage[2];

        // Stage 3
        pcout<< "3... " << std::flush;
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[3] = this->rk_stage[0];
        this->rk_stage[3] *= 1.0/3.0;
        this->rk_stage[3].add(2.0/3.0, this->rk_stage[2]);
        this->rk_stage[3].add(2.0/3.0*dt, this->solution_update);

        this->dg->solution = this->rk_stage[3];
        pcout<< "done." << std::endl;
    }

}

template <int dim, typename real>
void Explicit_ODESolver<dim,real>::allocate_ode_system ()
{
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->rk_stage.resize(4);
    for (int i=0; i<4; i++) {
        this->rk_stage[i].reinit(this->dg->solution);
    }
}
template <int dim, typename real>
void Implicit_ODESolver<dim,real>::allocate_ode_system ()
{
    const bool do_inverse_mass_matrix = false;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);
}

//template <int dim, typename real>
//std::shared_ptr<ODESolver<dim,real>> ODESolverFactory<dim,real>::create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
//{
//    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
//    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<Explicit_ODESolver<dim,real>>();
//    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<Implicit_ODESolver<dim,real>>();
//    else {
//        pcout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
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
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        pcout << "********************************************************************" << std::endl;
        pcout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
        pcout << "Solver type specified: " << ode_solver_type << std::endl;
        pcout << "Solver type possible: " << std::endl;
        pcout <<  ODEEnum::explicit_solver << std::endl;
        pcout <<  ODEEnum::implicit_solver << std::endl;
        pcout << "********************************************************************" << std::endl;
        std::abort();
        return nullptr;
    }
}

template class ODESolver<PHILIP_DIM, double>;
template class Explicit_ODESolver<PHILIP_DIM, double>;
template class Implicit_ODESolver<PHILIP_DIM, double>;
template class ODESolverFactory<PHILIP_DIM, double>;

} // ODE namespace
} // PHiLiP namespace
