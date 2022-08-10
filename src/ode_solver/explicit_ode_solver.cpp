#include "explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
ExplicitODESolver<dim,real,MeshType>::ExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , rk_order(this->ode_param.runge_kutta_order)
        {}

template <int dim, typename real, typename MeshType>
void ExplicitODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{  
    this->solution_update = this->dg->solution; //storing u_n
    
    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < rk_order; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau_a[i][j] != 0){
                this->rk_stage[i].add(this->butcher_tableau_a[i][j], this->rk_stage[j]);
            }
        } //sum(a_ij *k_j)
        
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)

        this->dg->solution = this->rk_stage[i];
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j))
        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }

    modify_time_step(dt);

    //assemble solution from stages
    for (int i = 0; i < rk_order; ++i){
        if (pseudotime){
            const double CFL = butcher_tableau_b[i] * dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
            this->solution_update.add(1.0, this->rk_stage[i]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau_b[i],this->rk_stage[i]); 
        }
    }
    this->dg->solution = this->solution_update; // u_np1 = u_n + dt* sum(k_i * b_i)

    ++(this->current_iteration);
    this->current_time += dt;
}

template <int dim, typename real, typename MeshType>
void ExplicitODESolver<dim,real,MeshType>::modify_time_step(real &/*dt*/)
{
    //do nothing
}

template <int dim, typename real, typename MeshType>
void ExplicitODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating inverse mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->rk_stage.resize(rk_order);
    for (int i=0; i<rk_order; i++) {
        this->rk_stage[i].reinit(this->dg->solution);
    }

    // Assigning butcher tableau
    this->butcher_tableau_a.reinit(rk_order,rk_order);
    this->butcher_tableau_b.reinit(rk_order);
    if (rk_order == 3){
        // RKSSP3 (RK-3 Strong-Stability-Preserving)
        const double butcher_tableau_a_values[9] = {0,0,0,1.0,0,0,0.25,0.25,0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
    } else if (rk_order == 4) {
        // Standard RK4
        const double butcher_tableau_a_values[16] = {0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0,1.0,0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
    } else if (rk_order == 1) {
        // Explicit Euler
        const double butcher_tableau_a_values[1] = {0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[1] = {1.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
    }
    else{
        this->pcout << "Invalid RK order" << std::endl;
        std::abort();
    }
}

template class ExplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ExplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ExplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODESolver namespace
} // PHiLiP namespace
