#include "low_storage_runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::LowStorageRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<LowStorageRKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , butcher_tableau(rk_tableau_input)
        , relaxation_runge_kutta(RRK_object_input)
{       epsilon[0] = 1.0;
        epsilon[1] = 1.0;
        epsilon[2] = 1.0; 
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::step_in_time (real dt, const bool /*pseudotime*/)
{
    storage_register_2.reinit(this->solution_update);
    storage_register_1.reinit(this->solution_update);

    storage_register_2*=0;
    storage_register_1 = this->dg->solution;
    storage_register_3 = storage_register_1;
    rhs = storage_register_1;
    if (is_3Sstarplus = true){
        storage_register_4 = storage_register_1;
        this->pcout << "s1 ";
        this->pcout << std::endl;
    } 

    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    //(void) pseudotime;
    double sum_delta = 0;

    //double atol = 0.001;
    //double rtol = 0.001;
    //double error = 0.0;
    //w = 0.0;

    for (int i = 1; i < n_rk_stages +1; i++ ){
        // storage_register_2 = storage_register_2 + delta[i-1] * storage_register_1;
        storage_register_2.add(this->butcher_tableau->get_delta(i-1) , storage_register_1);
        this->dg->solution = rhs;

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
        this->dg->assemble_residual();
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs);
        // storage_register_1 = gamma[i][0] * storage_register_1 + gamma[i][1] * storage_register_2 + gamma[i][2] * storage_register_3 + beta[i] * dt * rhs;
        storage_register_1 *= this->butcher_tableau->get_gamma(i, 0);
        storage_register_1.add(this->butcher_tableau->get_gamma(i, 1), storage_register_2);
        storage_register_1.add(this->butcher_tableau->get_gamma(i, 2), storage_register_3);
        rhs *= dt;
        storage_register_1.add(this->butcher_tableau->get_beta(i), rhs);
        if (this->butcher_tableau->get_b_hat(i) != 0.0){
            storage_register_4.add(this->butcher_tableau->get_b_hat(i-1), rhs);
        }
        //this->pcout << std::endl;
        // Check  bhat (i)
        // rhs = dt * f(S1)
        rhs = storage_register_1;
    }
   // std::abort();
    // storage_register_2 = (storage_register_2 + delta[m] * storage_register_1 + delta[m+1] * storage_register_3) / sum_delta;
    if (is_3Sstarplus = false){
        for (int i = 0; i < num_delta; i++){ //change n_rk_stages+2 to num_delta on this line
        sum_delta = sum_delta + this->butcher_tableau->get_delta(i);
        }
        storage_register_2.add(this->butcher_tableau->get_delta(n_rk_stages), storage_register_1);
        storage_register_2.add(this->butcher_tableau->get_delta(n_rk_stages+1), storage_register_3);
        storage_register_2 /= sum_delta;
        // u_hat = s2
    }


    // need to calculate rhs of s1
    else if (is_3Sstarplus = true){
        this->dg->solution = rhs;
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
        this->dg->assemble_residual();
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs);
        rhs *= dt;
        storage_register_4.add(this->butcher_tableau->get_b_hat(n_rk_stages), rhs);
        //this->pcout << " b_hat_fsal " << this->butcher_tableau->get_b_hat(n_rk_stages);

    }
    this->dg->solution = storage_register_1;

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
    
 /*   
    if (this->butcher_tableau->get_b_hat(1) == 0){
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_2.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_2.local_element(i)))), 2);
        }
    }
    else if (this->butcher_tableau->get_b_hat(1) != 0){
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_4.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_4.local_element(i)))), 2);
        }
    }
*/
    this->pcout << std::endl;

    ++(this->current_iteration);
    this->current_time += dt;
    this->pcout << " Time is: " << this->current_time <<std::endl;
    this->pcout << "dt" << dt;
}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
double LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::get_automatic_error_adaptive_step_size (real dt, const bool pseudotime)
{
    (void) pseudotime;
    //int q_hat = 2;
    //int k = q_hat +1;
    double beta_controller[3] = {0.70, -0.23, 0};
    double error = 0.0;
    w = 0.0;

    // error based step size 
    if (is_3Sstarplus = false){
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_2.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_2.local_element(i)))), 2);
        }
    }
    else if (is_3Sstarplus = true){
        // loop sums elements at each mpi processor
        for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < storage_register_1.local_size(); ++i) {
            error = storage_register_1.local_element(i) - storage_register_4.local_element(i);
            w = w + pow(error / (atol + rtol * std::max(std::abs(storage_register_1.local_element(i)), std::abs(storage_register_4.local_element(i)))), 2);
            /*
            this->pcout << "storage 1 " << storage_register_1.local_element(i);
            this->pcout << std::endl;
            this->pcout << "storage 4 " << storage_register_4.local_element(i);
            this->pcout << std::endl;
            this->pcout << "w " << w;
            this->pcout << std::endl;
            */
        }
    }
    this->pcout << std::endl;
    // sum over all elements
    w = dealii::Utilities::MPI::sum(w, this->mpi_communicator);
    w = pow(w / global_size, 0.5);
    this->pcout << std::endl;
    std::cout << "w2 " << w;
    this->pcout << std::endl;

    epsilon[2] = epsilon[1];
    epsilon[1] = epsilon[0];
    epsilon[0] = 1.0 / w;
    this->pcout << "eps0" << epsilon[0];
    this->pcout << std::endl;
    //dt = pow(epsilon[0], 1.0 * beta_controller[0]/k) * pow(epsilon[1], 1.0 * beta_controller[1]/k) * pow(epsilon[2], 1.0 * beta_controller[2]/k) * dt;
    dt = pow(epsilon[0], 1.0 * beta_controller[0]/rk_order) * pow(epsilon[1], 1.0 * beta_controller[1]/rk_order) * pow(epsilon[2], 1.0 * beta_controller[2]/rk_order) * dt;
    this->pcout << std::endl;
    this->pcout << "dt1" << dt;
    return dt;
}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::allocate_ode_system ()
{
    this->solution_update.reinit(this->dg->solution);
    storage_register_2.reinit(this->solution_update);
    storage_register_1.reinit(this->solution_update);
    storage_register_3.reinit(this->solution_update);
    storage_register_4.reinit(this->solution_update);

    atol = this->ode_param.atol;
    rtol = this->ode_param.rtol;

    global_size = dealii::Utilities::MPI::sum(storage_register_1.local_size(), this->mpi_communicator);
    this->pcout << "global size" << global_size;
    this->pcout << "Allocating ODE system..." << std::flush;
    this->solution_update.reinit(this->dg->right_hand_side);
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
        //RRK needs both mass matrix and inverse mass matrix
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_type = this->ode_param.ode_solver_type;
        if (ode_type == ODEEnum::rrk_explicit_solver){
            this->dg->evaluate_mass_matrices(false); // creates and stores global mass matrix
        }
    }
    
    this->pcout << std::endl;

    this->butcher_tableau->set_tableau();
}

template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,5, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
