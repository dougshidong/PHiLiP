#include <deal.II/distributed/solution_transfer.h>

#include "ode_solver.h"

#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace ODE {

double global_step = 1.0;

template <int dim, typename real, typename MeshType>
ODESolver<dim,real,MeshType>::ODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : current_time(0.0)
    , dg(dg_input)
    , all_parameters(dg->all_parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    n_refine = 0;
}

template <int dim, typename real, typename MeshType>
void ODESolver<dim,real,MeshType>::initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree)
{
    pcout << " ************************************************************************ " << std::endl;
    pcout << " Initializing DG with global polynomial degree = " << global_final_poly_degree << " by ramping from degree 0 ... " << std::endl;
    pcout << " ************************************************************************ " << std::endl;

    refine = false;
    for (unsigned int degree = 0; degree <= global_final_poly_degree; degree++) {
        if (degree == global_final_poly_degree) refine = true;
        pcout << " ************************************************************************ " << std::endl;
        pcout << " Ramping degree " << degree << " until p=" << global_final_poly_degree << std::endl;
        pcout << " ************************************************************************ " << std::endl;

        // Transfer solution to current degree.
        dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
        old_solution.update_ghost_values();
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
        dg->set_all_cells_fe_degree(degree);
        dg->allocate_system ();
        dg->solution.zero_out_ghosts();
        solution_transfer.interpolate(dg->solution);
        dg->solution.update_ghost_values();

        // Solve steady state problem.
        steady_state();
    }
}

template <int dim, typename real, typename MeshType>
bool ODESolver<dim,real,MeshType>::valid_initial_conditions () const
{
    for (const auto &sol : dg->solution) {
        if (sol == std::numeric_limits<real>::lowest()) {
            pcout << " User forgot to assign valid initial conditions. " << std::endl;
            return false;
        }
    }
    return true;
}

template <int dim, typename real, typename MeshType>
int ODESolver<dim,real,MeshType>::steady_state ()
{
    if (!valid_initial_conditions())
    {
        std::abort();
    }
    Parameters::ODESolverParam ode_param = ODESolver<dim,real,MeshType>::all_parameters->ode_solver_param;
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

    // Initial Courant-Friedrichs-Lax number
    const double initial_CFL = all_parameters->ode_solver_param.initial_time_step;
    CFL_factor = 1.0;

    auto initial_solution = dg->solution;

    double old_residual_norm = this->residual_norm; (void) old_residual_norm;

    int i_refine = 0;
    // Output initial solution
    int convergence_error = this->residual_norm > ode_param.nonlinear_steady_residual_tolerance;

    while (    convergence_error
            && this->residual_norm_decrease > ode_param.nonlinear_steady_residual_tolerance
            //&& update_norm             > ode_param.nonlinear_steady_residual_tolerance
            && this->current_iteration < ode_param.nonlinear_max_iterations
            && this->residual_norm     < 1e5
            && CFL_factor > 1e-2)
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
                  << " Residual norm (normalized) : " << this->residual_norm 
                  << " ( " << this->residual_norm / this->initial_residual_norm << " ) "
                  << std::endl;

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
            pcout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
        }

        double ramped_CFL = initial_CFL * CFL_factor;
        if (this->residual_norm_decrease < 1.0) {
            ramped_CFL *= pow((1.0-std::log10(this->residual_norm_decrease)*ode_param.time_step_factor_residual), ode_param.time_step_factor_residual_exp);
        }
        ramped_CFL = std::max(ramped_CFL,initial_CFL*CFL_factor);
        pcout << "Initial CFL = " << initial_CFL << ". Current CFL = " << ramped_CFL << std::endl;

        //if (this->residual_norm > 1e-9) this->dg->update_artificial_dissipation_discontinuity_sensor();
        //if (this->residual_norm > 1e-9) this->dg->update_artificial_dissipation_discontinuity_sensor();
        //if (this->residual_norm > 1e-9 || this->current_iteration > 50 ) this->dg->update_artificial_dissipation_discontinuity_sensor();
        //if ( this->current_iteration < 50 ) this->dg->update_artificial_dissipation_discontinuity_sensor();

        //if (this->residual_norm < 1e-12 || this->current_iteration > 20) {
        if (this->residual_norm < 1e-12) {
            this->dg->freeze_artificial_dissipation = true;
        } else {
            this->dg->freeze_artificial_dissipation = false;
        }
        //if (this->current_iteration % 1 == 0) {
        //    this->dg->freeze_artificial_dissipation = false;
        //} else {
        //    this->dg->freeze_artificial_dissipation = true;
        //}
        const bool pseudotime = true;
        step_in_time(ramped_CFL, pseudotime);

        this->dg->assemble_residual ();

        ++(this->current_iteration);

        if (ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }
        // if (this->residual_norm > old_residual_norm) {
        //     dg->refine_residual_based();
        //     allocate_ode_system ();
        // }
        //if ((current_iteration+1) % 10 == 0 || this->residual_norm > old_residual_norm) {
        //if (refine && global_step < 0.25 && this->current_iteration+1 > 10) {
        //if ( refine && (current_iteration+1) % 5 == 0 && this->residual_norm < 1e-11) {
        if ( refine && this->residual_norm < 1e-9 && i_refine < n_refine) {
            i_refine++;
            dg->refine_residual_based();
            allocate_ode_system ();
        }

        old_residual_norm = this->residual_norm;
        this->residual_norm = this->dg->get_residual_l2norm();
        this->residual_norm_decrease = this->residual_norm / this->initial_residual_norm;

        convergence_error = this->residual_norm > ode_param.nonlinear_steady_residual_tolerance
                            && this->residual_norm_decrease > ode_param.nonlinear_steady_residual_tolerance;
    }
    if (this->residual_norm > 1e5
        || std::isnan(this->residual_norm)
        || CFL_factor <= 1e-2)
    {
        this->dg->solution = initial_solution;

        if(CFL_factor <= 1e-2) this->dg->right_hand_side.add(1.0);
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

    return convergence_error;
}

template <int dim, typename real, typename MeshType>
int ODESolver<dim,real,MeshType>::advance_solution_time (double time_advance)
{
    Parameters::ODESolverParam ode_param = ODESolver<dim,real,MeshType>::all_parameters->ode_solver_param;

    const unsigned int number_of_time_steps = static_cast<int>(ceil(time_advance/ode_param.initial_time_step));
    const double constant_time_step = time_advance/number_of_time_steps;

    if (!valid_initial_conditions())
    {
        std::abort();
    }

    pcout
        << " Advancing solution by " << time_advance << " time units, using "
        << number_of_time_steps << " iterations of size dt=" << constant_time_step << " ... " << std::endl;
  //  allocate_ode_system ();

   // this->current_iteration = 0;
   pcout<<" curr iter "<<this->current_iteration<<std::endl;

    // Output initial solution
    if(this->current_iteration == 0){
pcout<<"allocating ode sys"<<std::endl;
        allocate_ode_system ();
        this->dg->output_results_vtk(this->current_iteration);
    }

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

    const bool pseudotime = false;
    step_in_time(constant_time_step, pseudotime);


    if (this->current_iteration%ode_param.print_iteration_modulo == 0) {
        this->dg->output_results_vtk(this->current_iteration);
    }

    if (ode_param.output_solution_vector_modulo > 0) {
        if (this->current_iteration % ode_param.output_solution_vector_modulo == 0) {
            for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
                solutions_table.template add_value(
                        "Time:" + std::to_string(this->current_iteration * constant_time_step),
                        this->dg->solution[i]);
            }
        }
    }
        ++(this->current_iteration);

       // this->dg->output_results_vtk(this->current_iteration);
    }

    if (ode_param.output_solution_vector_modulo > 0) {
        std::ofstream out_file("solutions_table.txt");
        solutions_table.write_text(out_file);
    }
    return 1;
}

template <int dim, typename real, typename MeshType>
void Implicit_ODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw
    Parameters::ODESolverParam ode_param = ODESolver<dim,real,MeshType>::all_parameters->ode_solver_param;

    this->dg->system_matrix *= -1.0;

    if (pseudotime) {
        const double CFL = dt;
        this->dg->time_scaled_mass_matrices(CFL);
        this->dg->add_time_scaled_mass_matrices();
    } else { 
        this->dg->add_mass_matrices(1.0/dt);
    }
    //(void) pseudotime;
    //this->dg->add_mass_matrices(1.0/dt);

    if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
        pcout << " Evaluating system update... " << std::endl;
    }

    solve_linear (
        this->dg->system_matrix,
        this->dg->right_hand_side,
        this->solution_update,
        this->ODESolver<dim,real,MeshType>::all_parameters->linear_solver_param);

    //this->dg->solution += this->solution_update;
    global_step = linesearch();

    this->update_norm = this->solution_update.l2_norm();
}

template <int dim, typename real, typename MeshType>
double Implicit_ODESolver<dim,real,MeshType>::linesearch ()
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
    pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
        pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }
    if (iline == maxline) {
        this->CFL_factor *= 0.5;
        pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        return 0.0;
    }

    if (iline == maxline) {
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        pcout << " Line search failed. Trying to step in the opposite direction. " << std::endl;
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
        //std::abort();
    }
    if (iline == maxline) {
        pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        this->CFL_factor *= 0.5;
    }

    return step_length;
}

template <int dim, typename real, typename MeshType>
void Explicit_ODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    // this->dg->assemble_residual (); // Not needed since it is called in the base class for time step
  //  this->current_time += dt;
    const int rk_order = 4;
    if (rk_order == 1) {
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->update_norm = this->solution_update.l2_norm();
        if (pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->dg->solution.add(1.0,this->solution_update);
        } else {
            this->dg->solution.add(dt,this->solution_update);
        }
    } else if (rk_order == 3) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;

        // Stage 1
        pcout<< "Stage 1... " << std::flush;
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[1] = this->rk_stage[0];
        //this->rk_stage[1].add(dt,this->solution_update);
        if (pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[1].add(1.0,this->solution_update);
        } else {
            this->rk_stage[1].add(dt,this->solution_update);
        }

        // Stage 2
        pcout<< "2... " << std::flush;
        this->dg->solution = this->rk_stage[1];
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[2] = this->rk_stage[0];
        this->rk_stage[2] *= 0.75;
        this->rk_stage[2].add(0.25, this->rk_stage[1]);
        //this->rk_stage[2].add(0.25*dt, this->solution_update);
        if (pseudotime) {
            const double CFL = 0.25*dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[2].add(1.0,this->solution_update);
        } else {
            this->rk_stage[2].add(0.25*dt,this->solution_update);
        }

        // Stage 3
        pcout<< "3... " << std::flush;
        this->dg->solution = this->rk_stage[2];
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);

        this->rk_stage[3] = this->rk_stage[0];
        this->rk_stage[3] *= 1.0/3.0;
        this->rk_stage[3].add(2.0/3.0, this->rk_stage[2]);
        //this->rk_stage[3].add(2.0/3.0*dt, this->solution_update);
        if (pseudotime) {
            const double CFL = (2.0/3.0)*dt;
            this->dg->time_scale_solution_update( this->solution_update, CFL );
            this->rk_stage[3].add(1.0,this->solution_update);
        } else {
            this->rk_stage[3].add((2.0/3.0)*dt,this->solution_update);
        }

        this->dg->solution = this->rk_stage[3];
        pcout<< "done." << std::endl;
    }
    else if (rk_order == 4) {
        // Stage 0
        this->rk_stage[0] = this->dg->solution;
        // Stage 1
       // pcout<< "Stage 1... " << std::flush;
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[1] = this->solution_update;
        this->rk_stage[1].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(0.5, this->rk_stage[1]);
        // Stage 2
       // pcout<< "2... " << std::flush;
        this->dg->set_current_time(this->current_time + dt/2.0);
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[2] = this->solution_update;
        this->rk_stage[2].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(0.5, this->rk_stage[2]);
        // Stage 3
       // pcout<< "3... " << std::flush;
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[3] = this->solution_update;
        this->rk_stage[3].operator*=(dt);
        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(1.0, this->rk_stage[3]);
        // Stage 4
       // pcout<< "4... " << std::flush;
        this->dg->set_current_time(this->current_time + dt);
        this->dg->assemble_residual ();
        this->dg->global_inverse_mass_matrix.vmult(this->solution_update, this->dg->right_hand_side);
        this->rk_stage[4] = this->solution_update;
        this->rk_stage[4].operator*=(dt);

        this->dg->solution=this->rk_stage[0];
        this->dg->solution.add(1.0/6.0, this->rk_stage[1]);
        this->dg->solution.add(1.0/3.0, this->rk_stage[2]);
        this->dg->solution.add(1.0/3.0, this->rk_stage[3]);
        this->dg->solution.add(1.0/6.0, this->rk_stage[4]);
       // pcout<< "done." << std::endl;
        this->current_time += dt;

    }

}

template <int dim, typename real, typename MeshType>
void Explicit_ODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    pcout << "Allocating ODE system and evaluating inverse mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->rk_stage.resize(5);
    for (int i=0; i<5; i++) {
        this->rk_stage[i].reinit(this->dg->solution);
    }
}
template <int dim, typename real, typename MeshType>
void Implicit_ODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);

}

//template <int dim, typename real, typename MeshType>
//std::shared_ptr<ODESolver<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
//{
//    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
//    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<Explicit_ODESolver<dim,real>>();
//    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<Implicit_ODESolver<dim,real>>();
//    else {
//        pcout << "Can't create ODE solver since explicit/implicit solver is not clear." << std::endl;
//        return nullptr;
//    }
//}
template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolver<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<Explicit_ODESolver<dim,real,MeshType>>(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<Implicit_ODESolver<dim,real,MeshType>>(dg_input);
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

// dealii::Triangulation<PHILIP_DIM>
template class ODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class Explicit_ODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class Implicit_ODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ODESolverFactory<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class ODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Explicit_ODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Implicit_ODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class ODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Explicit_ODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Implicit_ODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
