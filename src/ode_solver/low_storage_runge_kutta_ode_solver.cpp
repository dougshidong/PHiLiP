#include "low_storage_runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::LowStorageRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<LowStorageRKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input,RRK_object_input)
        , butcher_tableau(rk_tableau_input)
        , epsilon{1.0, 1.0, 1.0} 
        , atol(this->ode_param.atol)
        , rtol(this->ode_param.rtol)
        , rk_order(this->ode_param.rk_order)
        , is_3Sstarplus(this->ode_param.is_3Sstarplus)
        , num_delta(this->ode_param.num_delta)
        , beta1(this->ode_param.beta1)
        , beta2(this->ode_param.beta2)
        , beta3(this->ode_param.beta3)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::calculate_stage_solution(int istage, real /*dt*/, const bool pseudotime)
{
    if(istage == 0) prep_for_step_in_time();
    if(pseudotime == true){
        std::cout << "Error: pseudotime low-storage RK is not implemented." << std::endl;
        std::abort();
    }
    storage_register_2.add(this->butcher_tableau->get_delta(istage) , storage_register_1);
    this->dg->solution = rhs;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::calculate_stage_derivative (int istage, real dt)
{
    this->dg->assemble_residual();
    this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs);
    storage_register_1 *= this->butcher_tableau->get_gamma(istage+1, 0);
    storage_register_1.add(this->butcher_tableau->get_gamma(istage+1, 1), storage_register_2);
    storage_register_1.add(this->butcher_tableau->get_gamma(istage+1, 2), storage_register_3);
    rhs *= dt;
    storage_register_1.add(this->butcher_tableau->get_beta(istage+1), rhs);
    if (is_3Sstarplus == true){
        storage_register_4.add(this->butcher_tableau->get_b_hat(istage), rhs);
    }
    rhs = storage_register_1;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::sum_stages (real dt, const bool /*pseudotime*/)
{
    double sum_delta = 0.0;
    if (!is_3Sstarplus){
        for (int istage = 0; istage < num_delta; istage++){
            sum_delta += this->butcher_tableau->get_delta(istage);
        }
        storage_register_2.add(this->butcher_tableau->get_delta(n_rk_stages), storage_register_1);
        storage_register_2.add(this->butcher_tableau->get_delta(n_rk_stages+1), storage_register_3);
        storage_register_2 /= sum_delta;
    } else {
        this->dg->solution = rhs;
        // Apply limiter at every RK stage
        this->apply_limiter();
        this->dg->assemble_residual();
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs);
        rhs *= dt;
        storage_register_4.add(this->butcher_tableau->get_b_hat(n_rk_stages), rhs);       
    }

    this->solution_update = storage_register_1;

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
    (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Time is: " << this->current_time + dt <<std::endl;
        this->pcout << std::endl;
    }
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::apply_limiter ()
{
    // Apply limiter at every RK stage
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection);
    }
}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
real LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::adjust_time_step (real dt)
{  
    /*Empty function for now*/ 
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
double LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::get_automatic_error_adaptive_step_size (real dt, const bool /*pseudotime*/)
{
    double error = 0.0;
    w = 0.0;

    // error based step size 
    if (!is_3Sstarplus){ //False
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_2.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_2.local_element(i)))), 2);
        }
    } else { // True
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_4.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_4.local_element(i)))), 2);
        }
    }

    // sum over all elements
    w = dealii::Utilities::MPI::sum(w, this->mpi_communicator);
    w = pow(w / global_size, 0.5);
    epsilon[2] = epsilon[1];
    epsilon[1] = epsilon[0];
    epsilon[0] = 1.0 / w;
    dt = pow(epsilon[0], 1.0 * beta1/rk_order) * pow(epsilon[1], 1.0 * beta2/rk_order) * pow(epsilon[2], 1.0 * beta3/rk_order) * dt;
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
double LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::get_automatic_initial_step_size (real dt, const bool /*pseudotime*/)
{
    // h will store the starting step size
    double h0 = 0.0;
    double h1 = 0.0;

    // d estimates the derivative of the solution
    double d0 = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;

    /// Storage of the solution to calculate the initial time step
    dealii::LinearAlgebra::distributed::Vector<double> u_n;
    dealii::LinearAlgebra::distributed::Vector<double> rhs_initial;

    this->solution_update = this->dg->solution;
    u_n.reinit(this->solution_update);
    rhs_initial.reinit(this->solution_update);
    storage_register_1.reinit(this->solution_update);
    storage_register_1 = this->dg->solution;
    u_n = storage_register_1;
    rhs_initial = storage_register_1;

    d0 = u_n.linfty_norm();
    
    this->dg->solution = rhs_initial;
    this->dg->assemble_residual();
    this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs_initial);

    d1 = rhs_initial.linfty_norm();

    if (d0 < 1e-5 || d1 < 1e-5){
        h0 = 1e-6;
    }else{
        h0 = 0.01 * d0 / d1;
    }

    u_n.add(h0, rhs_initial);
    this->dg->solution = u_n;
    this->dg->assemble_residual();
    this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, u_n);
    
    // Calculate d2
    u_n.add(-1, rhs_initial);
    d2 = u_n.linfty_norm();
    d2 /= h0;

    if (std::max(d1, d2) <= 1e-15)
    {
        h1 = std::max(1e-6, h0 * 1e-3);
    }
    else
    {
        h1 = 0.01 / std::max(d1, d2);
    }

    dt = std::min(100 * h0, h1);
    this->dg->solution = storage_register_1;
    return dt;
}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::allocate_runge_kutta_system ()
{
    // Clear the rk_stage object for memory optimization
    this->rk_stage.clear();
    // Continue with allocating LSRK
    storage_register_1.reinit(this->dg->solution);
    this->solution_update = this->dg->solution; // This line needs to be included to properly run the prep for a step in time
    global_size = dealii::Utilities::MPI::sum(storage_register_1.local_size(), this->mpi_communicator);
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " use_inverse_mass_on_the_fly == false. Aborting!" << std::flush;
        std::abort();
        /*
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
        //RRK needs both mass matrix and inverse mass matrix
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_type = this->ode_param.ode_solver_type;
        if (ode_type == ODEEnum::rrk_explicit_solver){
            this->dg->evaluate_mass_matrices(false); // creates and stores global mass matrix
        }
        */
    }
    
    this->pcout << std::endl;

    this->butcher_tableau->set_tableau();
   
}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::prep_for_step_in_time()
{
    storage_register_1.reinit(this->solution_update);
    storage_register_2.reinit(this->solution_update);
    storage_register_1 = this->solution_update;
    storage_register_2 *= 0; // Unsure if this does anything as 2 should be zeroed from reinit function
    storage_register_3 = storage_register_1;
    rhs = storage_register_1;
    if (is_3Sstarplus == true){
        storage_register_4 = storage_register_1;
    } 
}
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,9, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,10, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,9, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,10, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,9, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,10, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
