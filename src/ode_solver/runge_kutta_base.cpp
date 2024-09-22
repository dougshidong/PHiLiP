#include "runge_kutta_base.h"

namespace PHiLiP {
namespace ODE {

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
            : ODESolverBase<dim,real,MeshType>(dg_input)
            , butcher_tableau(rk_tableau_input)
            , relaxation_runge_kutta(RRK_object_input)
            , solver(dg_input)
{}            


template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::step_in_time(real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    for (int i = 0; i < n_rk_stages; ++i){
        this->calculate_stages(i); // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(i) <implicit>
        this->apply_limiter(); // Might need to change for LSRK
        this->obtain_stage(i); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }
    this->modified_time_step();
    this->sum_stages(pseudotime); // u_np1 = u_n + dt* sum(k_i * b_i)
    this->dg->solution = this->solution_update; 
     // Calculate numerical entropy with FR correction. Does nothing if use has not selected param.
    this->FR_entropy_contribution_RRK_solver = relaxation_runge_kutta->compute_FR_entropy_contribution(dt, this->dg, this->rk_stage, true);
    this->apply_limiter();
    ++(this->current_iteration);
    this->current_time += dt;
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::allocate_ode_system()
{
    this->pcout << "Allocating ODE system..." << std::flush;
    this->solution_update.reinit(this->dg->right_hand_side);

    this->pcout << std::endl;
    
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

    this->allocate_runge_kutta_system();
}


template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // ODE namespace
} // PHiLiP namespace