#include "rrk_explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RRKExplicitODESolver<dim,real,MeshType>::RRKExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ExplicitODESolver<dim,real,MeshType>(dg_input)
        {}

template <int dim, typename real, typename MeshType>
void RRKExplicitODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{  

    if (pseudotime){
        this->pcout << "Pseudotime not implemented for RRK explicit. Aborting..." << std::endl;
        std::abort();
    }

    this->solution_update = this->dg->solution; //storing u_n
    
    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < this->rk_order; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau_a[i][j] != 0){
                this->rk_stage[i].add(this->butcher_tableau_a[i][j], this->rk_stage[j]);
            }
        } //sum(a_ij *k_j)
        
        this->rk_stage[i]*=dt; 
        
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)

        this->dg->solution = this->rk_stage[i];
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j))
        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }

    real relaxation_parameter = compute_relaxation_parameter_explicit();
    //assemble solution from stages
    for (int i = 0; i < this->rk_order; ++i){
       this->solution_update.add(
               dt * relaxation_parameter * this->butcher_tableau_b[i],
               this->rk_stage[i]); 
    }
    this->dg->solution = this->solution_update; // u_np1 = u_n +gamma*dt* sum(k_i * b_i)

    ++(this->current_iteration);
    this->current_time += relaxation_parameter * dt;
}

template <int dim, typename real, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_relaxation_parameter_explicit()
{
    double gamma = 1;
    double denominator = 0;
    double numerator = 0;
    for (int i = 0; i < this->rk_order; ++i){
        for (int j = 0; j < this->rk_order; ++j){
            real inner_product = compute_inner_product(this->rk_stage[i],this->rk_stage[j]);
            numerator += this->butcher_tableau_b[i] *this-> butcher_tableau_a[i][j] * inner_product; 
            denominator += this->butcher_tableau_b[i]*this->butcher_tableau_b[j] * inner_product;
        }
    }
    numerator *= 2;
    gamma = (denominator < 1E-8) ? 1 : numerator/denominator;
    return gamma;
}

template <int dim, typename real, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_inner_product (
        dealii::LinearAlgebra::distributed::Vector<double> stage_i,
        dealii::LinearAlgebra::distributed::Vector<double> stage_j
        )
{
    // Intention is to point to physics (mimic structure in flow_solver_cases/periodic_turbulence.cpp for converting to solution for general nodes) 
    // For now, only energy on collocated nodes is implemented.
    
    real inner_product = 0;
    for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
        inner_product += 1./(this->dg->global_inverse_mass_matrix.diag_element(i))
                         * stage_i[i] * stage_j[i];
    }
    return inner_product;
}

template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODESolver namespace
} // PHiLiP namespace
