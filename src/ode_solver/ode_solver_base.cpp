#include "ode_solver_base.h"

namespace PHiLiP {
namespace ODE{

template <int dim, typename real, typename MeshType>
ODESolverBase<dim,real,MeshType>::ODESolverBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : current_time(0.0)
        , current_iteration(0)
        , current_desired_time_for_output_solution_every_dt_time_intervals(0.0)
        , dg(dg_input)
        , all_parameters(dg->all_parameters)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
        , refine_mesh_in_ode_solver(true)
        {
            meshadaptation = std::make_unique<MeshAdaptation<dim,real,MeshType>>(dg);
        }

template <int dim, typename real, typename MeshType>
void ODESolverBase<dim,real,MeshType>::initialize_steady_polynomial_ramping (const unsigned int global_final_poly_degree)
{
    pcout << " ************************************************************************ " << std::endl;
    pcout << " Initializing DG with global polynomial degree = " << global_final_poly_degree << " by ramping from degree 0 ... " << std::endl;
    pcout << " ************************************************************************ " << std::endl;

    refine_mesh_in_ode_solver = false;
    for (unsigned int degree = 0; degree <= global_final_poly_degree; degree++) {
        if (degree == global_final_poly_degree) refine_mesh_in_ode_solver = true;
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
void ODESolverBase<dim,real,MeshType>::valid_initial_conditions () const
{
    for (const auto &sol : dg->solution) {
        if (sol == std::numeric_limits<real>::lowest()) {
            throw std::invalid_argument(" User forgot to assign valid initial conditions. ");
        }
    }
}

template <int dim, typename real, typename MeshType>
int ODESolverBase<dim,real,MeshType>::steady_state ()
{
    try {
        valid_initial_conditions();
    }
    catch( const std::invalid_argument& e ) {
        std::abort();
    }

    Parameters::ODESolverParam ode_param = ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param;
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

        if (this->residual_norm < 1e-12) {
            this->dg->freeze_artificial_dissipation = true;
        } else {
            this->dg->freeze_artificial_dissipation = false;
        }

        const bool pseudotime = true;
        step_in_time(ramped_CFL, pseudotime);

        this->dg->assemble_residual ();

        if (ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }
        
        if ((this->residual_norm < meshadaptation->critical_residual) 
            && (refine_mesh_in_ode_solver) 
            && (meshadaptation->current_refinement_cycle < meshadaptation->total_refinement_cycles))
        {
            meshadaptation->adapt_mesh();
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

    if (ode_param.output_solution_vector_modulo > 0) {
        for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
            solutions_table.add_value(
                    "Steady-state solution:",
                    this->dg->solution[i]);
        }
        solutions_table.set_precision("Steady-state solution:", 16);
        std::ofstream out_file(ode_param.solutions_table_filename + ".txt");
        solutions_table.write_text(out_file);
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
int ODESolverBase<dim,real,MeshType>::advance_solution_time (double time_advance)
{
    Parameters::ODESolverParam ode_param = ODESolverBase<dim,real,MeshType>::all_parameters->ode_solver_param;

   // const unsigned int number_of_time_steps = static_cast<int>(ceil(time_advance/ode_param.initial_time_step));
    const unsigned int number_of_time_steps = (!this->all_parameters->use_energy) ? static_cast<int>(ceil(time_advance/ode_param.initial_time_step)) : this->current_iteration+1;
   // const double constant_time_step = time_advance/number_of_time_steps;
    const double constant_time_step = time_advance/static_cast<int>(ceil(time_advance/ode_param.initial_time_step));

    try {
        valid_initial_conditions();
    }
    catch( const std::invalid_argument& e ) {
        std::abort();
    }

    pcout
            << " Advancing solution by " << time_advance << " time units, using "
            << number_of_time_steps << " iterations of size dt=" << constant_time_step << " ... " << std::endl;
    allocate_ode_system ();

    if(!this->all_parameters->use_energy)
        this->current_iteration = 0;

    if (ode_param.output_solution_every_x_steps >= 0) {
        this->dg->output_results_vtk(this->current_iteration);  
    } else if (ode_param.output_solution_every_dt_time_intervals > 0.0) {
        this->dg->output_results_vtk(this->current_iteration);
        this->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
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

        if ((ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%ode_param.print_iteration_modulo) == 0 ) {
            pcout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
        }

        const bool pseudotime = false;
        step_in_time(constant_time_step, pseudotime);

        if (ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        } else if(ode_param.output_solution_every_dt_time_intervals > 0.0) {
            const bool is_output_time = ((this->current_time <= this->current_desired_time_for_output_solution_every_dt_time_intervals) && 
                                         ((this->current_time + constant_time_step) > this->current_desired_time_for_output_solution_every_dt_time_intervals));
            if (is_output_time) {
                const int file_number = this->current_desired_time_for_output_solution_every_dt_time_intervals / ode_param.output_solution_every_dt_time_intervals;
                this->dg->output_results_vtk(file_number);
                this->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
            }
        }

        if (ode_param.output_solution_vector_modulo > 0) {
            if (this->current_iteration % ode_param.output_solution_vector_modulo == 0) {
                for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
                    solutions_table.add_value(
                            "Time:" + std::to_string(this->current_time),
                            this->dg->solution[i]);
                }
                solutions_table.set_precision("Time:" + std::to_string(this->current_time), 16);
            }
        }
    }

    if (ode_param.output_solution_vector_modulo > 0) {
        std::ofstream out_file(ode_param.solutions_table_filename + ".txt");
        solutions_table.write_text(out_file);
    }
    return 1;
}

template class ODESolverBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ODESolverBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ODESolverBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
