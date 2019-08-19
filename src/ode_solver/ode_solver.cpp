#include "ode_solver.h"
#include "linear_solver/linear_solver.h"
#include <deal.II/distributed/solution_transfer.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real>
void ODESolver<dim,real>::initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree)
{
    std::cout << " ************************************************************************ " << std::endl;
    std::cout << " Initializing DG with global polynomial degree = " << global_final_poly_degree << " by ramping from degree 0 ... " << std::endl;
    std::cout << " ************************************************************************ " << std::endl;

    for (unsigned int degree = 0; degree <= global_final_poly_degree; degree++) {
        std::cout << " ************************************************************************ " << std::endl;
        std::cout << " Ramping degree " << degree << " until p=" << global_final_poly_degree << std::endl;
        std::cout << " ************************************************************************ " << std::endl;

        dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
        old_solution.update_ghost_values();

        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(old_solution);

        dg->set_all_cells_fe_degree(degree);
        //dg->triangulation->execute_coarsening_and_refinement();
        // Required even if no mesh refinement takes place
        //dg->triangulation->execute_coarsening_and_refinement();
        //dg->triangulation->refine_global (1);
        dg->allocate_system ();

        old_solution.print(std::cout);
        dg->solution.zero_out_ghosts();
        solution_transfer.interpolate(dg->solution);
        dg->solution.update_ghost_values();
        dg->solution.print(std::cout);

        //dealii::LinearAlgebra::distributed::Vector<double> new_solution(dg->locally_owned_dofs, MPI_COMM_WORLD);
        //new_solution.zero_out_ghosts();
        //solution_transfer.interpolate(new_solution);
        //new_solution.update_ghost_values();
        //new_solution.print(std::cout);

        steady_state();
    }
}

template <int dim, typename real>
int ODESolver<dim,real>::steady_state ()
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;
    std::cout << " Performing steady state analysis... " << std::endl;
    allocate_ode_system ();

    this->residual_norm = 1;
    this->residual_norm_decrease = 1; // Always do at least 1 iteration
    update_norm = 1; // Always do at least 1 iteration
    this->current_iteration = 0;

    this->dg->output_results_vtk(this->current_iteration);

    std::cout << " Evaluating right-hand side and setting system_matrix to Jacobian before starting iterations... " << std::endl;
    this->dg->assemble_residual ();
    initial_residual_norm = this->dg->get_residual_l2norm();

    // Output initial solution
    while (    this->residual_norm     > ode_param.nonlinear_steady_residual_tolerance 
            && this->residual_norm_decrease > ode_param.nonlinear_steady_residual_tolerance 
            && update_norm             > ode_param.nonlinear_steady_residual_tolerance 
            && this->current_iteration < ode_param.nonlinear_max_iterations )
    {
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " ********************************************************** "
                  << std::endl
                  << " Nonlinear iteration: " << this->current_iteration + 1
                  << " residual norm: " << this->residual_norm
                  << std::endl;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;

        this->dg->assemble_residual ();
        // for (unsigned int i=0; i<dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++) {
        //     if (i==dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
        //         std::cout << "residual MPI process " << i << std::endl;
        //         this->dg->right_hand_side.print(std::cout);
        //         std::cout << std::endl;
        //     }
        //     MPI_Barrier(MPI_COMM_WORLD);
        // }
        this->residual_norm = this->dg->get_residual_l2norm();
        this->residual_norm_decrease = this->residual_norm / this->initial_residual_norm;

        double dt = ode_param.initial_time_step;
        dt *= pow((1.0-std::log10(this->residual_norm_decrease)*ode_param.time_step_factor_residual), ode_param.time_step_factor_residual_exp);
        //const double decrease_log = (1.0-std::log10(this->residual_norm_decrease));
        //dt *= dt*pow(10, decrease_log);
        std::cout << "Time step = " << dt << std::endl;

        step_in_time(dt);

        ++(this->current_iteration);

        if (ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }

    }
    return 1;
}

template <int dim, typename real>
int ODESolver<dim,real>::advance_solution_time (double time_advance)
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real>::all_parameters->ode_solver_param;

    const unsigned int number_of_time_steps = static_cast<int>(ceil(time_advance/ode_param.initial_time_step));
    const double constant_time_step = time_advance/number_of_time_steps;

    std::cout
        << " Advancing solution by " << time_advance << " time units, using "
        << number_of_time_steps << " iterations of size dt=" << constant_time_step << " ... " << std::endl;
    allocate_ode_system ();

    this->current_iteration = 0;

    // Output initial solution
    this->dg->output_results_vtk(this->current_iteration);

    while (this->current_iteration < number_of_time_steps)
    {
        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " ********************************************************** "
                  << std::endl
                  << " Iteration: " << this->current_iteration + 1
                  << " out of: " << number_of_time_steps
                  << std::endl;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
        std::cout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;

        step_in_time(constant_time_step);

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

    this->dg->add_mass_matrices(1.0/dt);

    if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%ode_param.print_iteration_modulo) == 0 )
    std::cout << " Evaluating system update... " << std::endl;

    solve_linear (
        this->dg->system_matrix,
        this->dg->right_hand_side, 
        this->solution_update,
        this->ODESolver<dim,real>::all_parameters->linear_solver_param);

    this->dg->solution += this->solution_update;

    this->update_norm = this->solution_update.l2_norm();
}

template <int dim, typename real>
void Explicit_ODESolver<dim,real>::step_in_time (real dt)
{
    this->dg->assemble_residual ();
    this->current_time += dt;
    const int rk_order = 3;
    if (rk_order == 1) {
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->update_norm = this->solution_update.l2_norm();
        this->dg->solution.add(dt,this->solution_update);
    } else if (rk_order == 3) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;

        // Stage 1
        std::cout<< "Stage 1... ";
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[1] = this->rk_stage[0];
        this->rk_stage[1].add(dt,this->solution_update);

        this->dg->solution = this->rk_stage[1];

        // Stage 2
        std::cout<< "2... ";
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[2] = this->rk_stage[0];
        this->rk_stage[2] *= 0.75;
        this->rk_stage[2].add(0.25, this->rk_stage[1]);
        this->rk_stage[2].add(0.25*dt, this->solution_update);

        this->dg->solution = this->rk_stage[2];

        // Stage 3
        std::cout<< "3... ";
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[3] = this->rk_stage[0];
        this->rk_stage[3] *= 1.0/3.0;
        this->rk_stage[3].add(2.0/3.0, this->rk_stage[2]);
        this->rk_stage[3].add(2.0/3.0*dt, this->solution_update);

        this->dg->solution = this->rk_stage[3];
        std::cout<< "done." << std::endl;
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
