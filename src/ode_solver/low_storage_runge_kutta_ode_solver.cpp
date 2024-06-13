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
        , solver(dg_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    (void) pseudotime;

    //const double gamma[6][3] = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-0.497531095840104, 1.384996869124138, 0.0}, {1.010070514199942, 3.878155713328178, 0.0}, {-3.196559004608766,-2.324512951813145, 1.642598936063715}, {1.717835630267259, -0.514633322274467, 0.188295940828347}};
    //double beta[6] = {0.0, 0.075152045700771, 0.211361016946069, 1.100713347634329, 0.728537814675568, 0.393172889823198};
    //double delta[7] = {1.0, 0.081252332929194, -1.083849060586449, -1.096110881845602, 2.859440022030827, -0.655568367959557, -0.194421504490852};
    // double beta_controller[3] = {0.70, -0.40, 0.0}; // PI34

    dealii::LinearAlgebra::distributed::Vector<double> s2;
    s2.reinit(this->solution_update);
    s2*=0;
    dealii::LinearAlgebra::distributed::Vector<double> s3;
    s3.reinit(this->solution_update);
    s3 = this->dg->solution;
    dealii::LinearAlgebra::distributed::Vector<double> s1 = s3;
    //rhs.reinit(this->solution_update);
    //rhs = s3;
    //dealii::LinearAlgebra::distributed::Vector<double> u_hat = s2;
    //u_hat.reinit(this->solution_update);
    //u_hat = s2;

    dealii::LinearAlgebra::distributed::Vector<double> rhs = s1;

    int m = 5;
    int q_hat = 3;
    int k = q_hat +1;
    double sum_delta = 0;
    double error = 0.0;
    double w = 0.0;
    double atol = 0.001;
    double rtol = 0.001;
    double epsilon[3] = {1, 1, 1};
    double beta_controller[3] = {0.70, -0.40, 0};


    for (int i = 1; i < m+1; i++ ){
        // s2 = s2 + delta[i-1] * s1;
        s2.add(this->butcher_tableau->get_delta(i-1) , s1);
        this->dg->solution = rhs;
        this->dg->assemble_residual();
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, rhs);
        // s1 = gamma[i][0] * s1 + gamma[i][1] * s2 + gamma[i][2] * s3 + beta[i] * dt * rhs;
        s1 *= this->butcher_tableau->get_gamma(i, 0);
        s1.add(this->butcher_tableau->get_gamma(i, 1), s2);
        s1.add(this->butcher_tableau->get_gamma(i, 2), s3);
        rhs *= dt;
        s1.add(this->butcher_tableau->get_beta(i), rhs);
        rhs = s1;

    }
    for (int i = 0; i<m+2; i++){
        sum_delta = sum_delta + this->butcher_tableau->get_delta(i);
    }
    // s2 = (s2 + delta[m] * s1 + delta[m+1] * s3) / sum_delta;
    s2.add(this->butcher_tableau->get_delta(m), s1);
    s2.add(this->butcher_tableau->get_delta(m+1), s3);
    s2 /= sum_delta;

    this->dg->solution = s1;
    //u_hat = s2;

    // error based step size 
    
    for (dealii::LinearAlgebra::distributed::Vector<double>::size_type i = 0; i < s1.local_size(); ++i) {
        error = s1.local_element(i) - s2.local_element(i);
        w = w + pow(error / (atol + rtol * std::max(std::abs(s1.local_element(i)), std::abs(s2.local_element(i)))), 2);
    }
    w = pow(w / s1.local_size(), 1/2);
    epsilon[2] = epsilon[1];
    epsilon[1] = epsilon[0];
    epsilon[0] = 1 / w;
    dt = pow(epsilon[0], beta_controller[0]/k) * pow(epsilon[1], beta_controller[1]/k) * pow(epsilon[2], beta_controller[2]/k) * dt;
    

    ++(this->current_iteration);
    this->current_time += dt;
    this->pcout << "Time is: " << this->current_time <<std::endl;
    this->pcout << "dt" << dt;

}


template <int dim, typename real, int n_rk_stages, typename MeshType> 
void LowStorageRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_ode_system ()
{
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
/*
    this->rk_stage.resize(n_rk_stages);
    for (int i=0; i<n_rk_stages; ++i) {
        this->rk_stage[i].reinit(this->dg->solution);
    }

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int i=0; i<n_rk_stages; ++i) {
        if (this->butcher_tableau->get_a(i,i)==0.0)     this->butcher_tableau_aii_is_zero[i] = true;
    
    }
    */
}

template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class LowStorageRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
